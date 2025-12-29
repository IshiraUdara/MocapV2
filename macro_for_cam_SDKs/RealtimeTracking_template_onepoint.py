import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
import queue
import socket

import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure.k4a import Image, _k4a

from lib.Helpers import find_point_correspondance_and_object_points, get_extrinsics
from lib.ImageOperations import _find_dot

import random
# Optional Kinect v2 (Kinect for Xbox One) support
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
except Exception:
    PyKinectRuntime = None
    PyKinectV2 = None

import EZVIZ_camera_feedToMMAP as main_camera

running = threading.Event()
running.set()

def _is_nonempty(x):
    """Return True if x is not None and contains at least one element.
    Safe for Python sequences and NumPy arrays (avoids ambiguous truth-value).
    """
    if x is None:
        return False
    # NumPy arrays have .size
    if hasattr(x, "size"):
        try:
            return int(x.size) > 0
        except Exception:
            pass
    try:
        return len(x) > 0
    except Exception:
        # fallback: assume non-None object is non-empty
        return True

camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")
if camera_count < 2:
    print(f"[Init] Need 2 camera extrinsics, got {camera_count}. 3D fusion will output zeros.")
else:
    # optional: print camera names/order
    try:
        names = [cp.get("name", f"cam{i}") for i, cp in enumerate(camera_poses)]
        print(f"[Init] Loaded extrinsics for: {names}")
    except Exception:
        print("[Init] Loaded extrinsics, but could not list names.")

# Lightweight queues to pass detections (keep only latest)
data_queue_azure = queue.Queue(maxsize=4)
data_queue_kv2 = queue.Queue(maxsize=4)


def _drain_latest(q):
    """Return the newest item from queue or None if empty."""
    item = None
    try:
        while True:
            item = q.get_nowait()
    except queue.Empty:
        return item


def track_points_azure(kinect, out_queue: queue.Queue, preview=False):
    """Acquire IR from Azure Kinect DK, preprocess, detect dot, push detections."""
    try:
        serial = kinect.get_serialnum()
        win = f'Azure IR - {serial}'
        if preview:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        time.sleep(1.0)
        while running.is_set():
            try:
                capture = kinect.get_capture()
                if capture is None:
                    time.sleep(0.001)
                    continue

                ir_handle = _k4a.k4a_capture_get_ir_image(capture.handle if hasattr(capture, 'handle') else capture)
                if not ir_handle:
                    continue

                try:
                    w = _k4a.k4a_image_get_width_pixels(ir_handle)
                    h = _k4a.k4a_image_get_height_pixels(ir_handle)
                    size = _k4a.k4a_image_get_size(ir_handle)
                    if w <= 0 or h <= 0 or size <= 0:
                        continue

                    arr = Image(ir_handle).to_numpy()
                    ir_image = arr if not isinstance(arr, tuple) else arr[1]
                    if ir_image is None:
                        continue

                    # robust preprocess -> uint8
                    if ir_image.dtype == np.uint16:
                        gray8 = _azure_preprocess_ir_u16(ir_image)
                    else:
                        gray8 = cv2.normalize(ir_image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                finally:
                    _k4a.k4a_image_release(ir_handle)

                processed, detected = _azure_detect_dot(gray8, bright_dot=True, max_points=2)
                # enqueue only valid detections with a timestamp
                is_valid = (
                    _is_nonempty(detected)
                    and detected[0][0] is not None
                    and detected[0][1] is not None
                )
                
                if is_valid:
                    try:
                        if out_queue.full():
                            out_queue.get_nowait()
                        out_queue.put_nowait((time.time(), detected))  # (ts, [[x,y]])
                    except queue.Full:
                        pass

                if preview:
                    cv2.imshow(win, processed)
                    if cv2.waitKey(1) & 0xFF == 27:
                        running.clear()
                        break

            except Exception as e:
                print(f"Azure loop error: {e}")
                time.sleep(0.01)
                continue

        if preview:
            cv2.destroyWindow(win)
        return True

    except Exception as e:
        print(f"Azure tracker error: {e}")
        return False


def track_points_kinect_v2(kinect_runtime, out_queue: queue.Queue, preview=False):
    """Acquire IR from Kinect v2 runtime, detect dots and push detections."""
    if PyKinectRuntime is None or PyKinectV2 is None:
        print("PyKinect2 not available; Kinect v2 disabled.")
        return False

    try:
        win = 'Kinect v2 IR'
        if preview:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # try to read frame descriptor
        try:
            ir_desc = kinect_runtime.infrared_frame_desc
            h, w = ir_desc.Height, ir_desc.Width
        except Exception:
            h, w = 424, 512

        time.sleep(1.0)
        while running.is_set():
            try:
                ir_frame = kinect_runtime.get_last_infrared_frame()
                if ir_frame is None:
                    time.sleep(0.001)
                    continue

                try:
                    ir_image = np.array(ir_frame, dtype=np.uint16).reshape((h, w))
                except Exception:
                    ir_image = ir_frame.reshape((h, w)).astype(np.uint16)

                # PREPROCESS AND DETECT (replace >>8 + _find_dot)
                gray8 = _kv2_preprocess_ir_u16(ir_image)
                processed, detected = _kv2_detect_dot(gray8, bright_dot=True, max_points=2)
                
                # enqueue only valid detections with a timestamp
                is_valid = (
                    _is_nonempty(detected)
                    and detected[0][0] is not None
                    and detected[0][1] is not None
                )
                if is_valid:
                    try:
                        if out_queue.full():
                            out_queue.get_nowait()
                        out_queue.put_nowait((time.time(), detected))  # (ts, [[x,y]])
                    except queue.Full:
                        pass

                if preview:
                    cv2.imshow(win, processed)
                    if cv2.waitKey(1) & 0xFF == 27:
                        running.clear()
                        break
            except Exception as e:
                print(f"Kinect v2 loop error: {e}")
                time.sleep(0.01)
                continue

        if preview:
            cv2.destroyWindow(win)
        return True

    except Exception as e:
        print(f"Kinect v2 tracker error: {e}")
        return False


def _select_best_3d_point(points3d, cluster_threshold=0.06):
    """Cluster candidate 3D points and return centroid of largest cluster."""
    # use safe emptiness check to avoid "truth value of an array is ambiguous"
    if not _is_nonempty(points3d):
        return None
    pts = np.array(points3d, dtype=np.float32)
    if pts.shape[0] == 1:
        return pts[0].tolist()
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    N = pts.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            neigh = np.where(d[j] < cluster_threshold)[0]
            for n in neigh:
                if not visited[n]:
                    visited[n] = True
                    stack.append(n)
        clusters.append(comp)
    clusters.sort(key=lambda c: len(c), reverse=True)
    best = clusters[0]
    centroid = pts[best].mean(axis=0)
    return centroid.tolist()

def _kv2_preprocess_ir_u16(ir_u16: np.ndarray) -> np.ndarray:
    """Normalize Kinect v2 IR (uint16) to contrasty uint8."""
    if ir_u16 is None:
        return None
    a = ir_u16.astype(np.uint16)
    # clip very bright speculars to reduce false blobs
    hi = np.percentile(a, 99.5)
    a = np.clip(a, 0, hi).astype(np.uint16)
    a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # boost local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(a8)

def _kv2_detect_dot(gray8: np.ndarray, bright_dot: bool = True, max_points=2):
    """Detect up to two circular dots in KV2 IR using SimpleBlobDetector."""
    if gray8 is None or gray8.ndim != 2:
        return gray8, [[None, None]] * max_points

    h, w = gray8.shape
    area_scale = (w * h) / float(512 * 424)
    min_area = int(20 * area_scale)
    max_area = int(2200 * area_scale)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255 if bright_dot else 0
    p99 = int(np.percentile(gray8, 99))
    params.minThreshold = max(170, p99 - 15)
    params.maxThreshold = 255
    params.thresholdStep = 5

    params.filterByArea = True
    params.minArea = max(8, min_area)
    params.maxArea = max_area

    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(gray8)

    disp = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    if not kps:
        return disp, [[None, None]] * max_points
    # Sort by size, take up to max_points
    kps = sorted(kps, key=lambda kp: kp.size, reverse=True)[:max_points]
    points = []
    for k in kps:
        cx, cy = int(k.pt[0]), int(k.pt[1])
        cv2.circle(disp, (cx, cy), 6, (0, 0, 255), 2)
        points.append([float(cx), float(cy)])
    # Pad if less than max_points
    while len(points) < max_points:
        points.append([None, None])
    return disp, points

def _azure_preprocess_ir_u16(ir_u16: np.ndarray) -> np.ndarray:
    """Normalize Azure IR (uint16) to contrasty uint8."""
    if ir_u16 is None:
        return None
    a = ir_u16.astype(np.uint16)
    # clip extreme highs to suppress specular bloom
    hi = np.percentile(a, 99.7)
    a = np.clip(a, 0, hi).astype(np.uint16)
    # normalize -> uint8
    a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # denoise and boost local contrast
    a8 = cv2.medianBlur(a8, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a8 = clahe.apply(a8)
    return a8

def _azure_detect_dot(gray8: np.ndarray, bright_dot: bool = True, max_points=2):
    """Detect up to two bright circular dots in Azure IR using SimpleBlobDetector."""
    if gray8 is None or gray8.ndim != 2:
        return gray8, [[None, None]] * max_points

    h, w = gray8.shape
    area_scale = (w * h) / float(512 * 512)
    min_area = int(18 * area_scale)
    max_area = int(1800 * area_scale)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255 if bright_dot else 0

    p99 = int(np.percentile(gray8, 99))
    params.minThreshold = max(160, p99 - 20)
    params.maxThreshold = 255
    params.thresholdStep = 5

    params.filterByArea = True
    params.minArea = max(5, min_area)
    params.maxArea = max_area

    params.filterByCircularity = True
    params.minCircularity = 0.65
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.85

    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(gray8)

    disp = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    if not kps:
        return disp, [[None, None]] * max_points
    # Sort by size, take up to max_points
    kps = sorted(kps, key=lambda kp: kp.size, reverse=True)[:max_points]
    points = []
    for k in kps:
        cx, cy = int(k.pt[0]), int(k.pt[1])
        cv2.circle(disp, (cx, cy), 6, (0, 0, 255), 2)
        points.append([float(cx), float(cy)])
    while len(points) < max_points:
        points.append([None, None])
    return disp, points

# Add helpers for single-camera fallback (ray-plane)
def _extract_K_R_t(cp: dict):
    """Best-effort extraction without using boolean 'or' on numpy arrays."""
    if cp is None:
        return None, None, None

    def first(keys):
        for k in keys:
            if k in cp and cp[k] is not None:
                return cp[k]
        return None

    # Intrinsics
    K = first(["camera_matrix", "intrinsic_matrix", "K"])
    if K is not None:
        K = np.array(K, dtype=np.float64)

    # Rotation/translation
    Rv = first(["R", "rotation", "rotation_matrix"])
    tv = first(["t", "translation", "T"])

    # If missing, try 4x4 extrinsic matrix
    Em = first(["extrinsic_matrix", "extrinsics", "Rt"])
    if (Rv is None or tv is None) and Em is not None:
        E = np.array(Em, dtype=np.float64)
        if E.shape == (4, 4):
            if Rv is None:
                Rv = E[:3, :3]
            if tv is None:
                tv = E[:3, 3]

    if Rv is not None:
        Rv = np.array(Rv, dtype=np.float64)
    if tv is not None:
        tv = np.array(tv, dtype=np.float64).reshape(3)

    return K, Rv, tv

def _get_floor_plane(extrinsics: list):
    """Find a plane {normal:[nx,ny,nz], d} in JSON; fallback to z=0."""
    plane = None
    for cp in extrinsics or []:
        p = cp.get("floor_plane") or cp.get("ground_plane") or cp.get("plane")
        if isinstance(p, dict) and "normal" in p and "d" in p:
            plane = {"normal": np.array(p["normal"], dtype=np.float64).reshape(3),
                     "d": float(p["d"])}
            break
    if plane is None:
        plane = {"normal": np.array([0.0, 0.0, 1.0]), "d": 0.0}
    return plane

def _ray_from_pixel(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv: np.ndarray):
    """Return world-space ray (origin C, dir) from pixel uv. World->cam: Xc=R Xw + t."""
    Kinv = np.linalg.inv(K)
    ray_cam = Kinv @ np.array([float(uv[0]), float(uv[1]), 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    Rt = R.T
    C = -Rt @ t
    dir_w = Rt @ ray_cam
    dir_w /= np.linalg.norm(dir_w)
    return C, dir_w

def _intersect_ray_plane(C: np.ndarray, dir_w: np.ndarray, plane: dict):
    """Intersect with n·X + d = 0; return None if parallel/behind."""
    n = plane["normal"]; d = plane["d"]
    denom = float(n.dot(dir_w))
    if abs(denom) < 1e-8:
        return None
    t = -(n.dot(C) + d) / denom
    if t <= 0:
        return None
    X = C + t * dir_w
    return X

def track(out_queue_azure: queue.Queue, out_queue_kv2: queue.Queue, stream=True):
    """Fuse detections from both cameras, triangulate and send best single 3D point.
       Server-only mode: Python acts as TCP server; Unity should connect as client.

       Communication matches test.py style: plain UTF-8 lines sent over TCP.
       Each line: "x y z" (space separated, meters), newline-delimited.
    """
    global camera_poses
    print("Fused tracking started (server mode)")
    # --- WARMUP ---
    print("Warming up cameras for 3 seconds...")
    time.sleep(3)
    print("Warmup complete. Starting tracking.")

    # Prepare single-camera fallback using cam0 (Azure) if available
    plane = _get_floor_plane(camera_poses or [])
    K0 = R0 = t0 = None
   
    if _is_nonempty(camera_poses):
        K0, R0, t0 = _extract_K_R_t(camera_poses[0])
        if K0 is None or R0 is None or t0 is None:
            print("[Fallback] Missing Azure K/R/t in extrinsics; single-camera fallback disabled.")
        else:
            print("[Fallback] Single-camera floor-plane fallback enabled.")

    server = None
    conn = None

    if stream:
        HOST = "127.0.0.1"   # Python server address (Unity client should connect here)
        PORT = 5005           # match test.py default

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((HOST, PORT))
        except Exception as e:
            print(f"Failed to bind server on {HOST}:{PORT}: {e}")
            # try ephemeral port
            server.bind((HOST, 0))
            PORT = server.getsockname()[1]
            print(f"Bound to ephemeral port {PORT}")
        server.listen(1)
        print(f"Waiting for Unity client to connect on {HOST}:{PORT} (server mode)...")

    else:
        server = None
        conn = None

    # cache last valid per-camera detections
    last_az_ts, last_az_pts = 0.0, None
    last_kv2_ts, last_kv2_pts = 0.0, None
    STALE_S = 1.0  # was 0.25; allow up to 1s skew between detections

    while running.is_set():
        try:
            # Accept client if needed
            if stream and conn is None:
                try:
                    conn, addr = server.accept()
                    print(f"Unity connected from {addr}")
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception as e:
                    print(f"Accept failed: {e}")
                    time.sleep(0.1)
                    continue

            # pull newest items, update caches only when new data present
            az_item = _drain_latest(out_queue_azure)
            if az_item is not None:  # (ts, [[x,y]])
                t, pts = az_item
                last_az_ts, last_az_pts = t, pts

            kv2_item = _drain_latest(out_queue_kv2)
            if kv2_item is not None:
                t, pts = kv2_item
                last_kv2_ts, last_kv2_pts = t, pts

            now = time.time()
            have_az = last_az_pts is not None and (now - last_az_ts) <= STALE_S
            have_kv2 = last_kv2_pts is not None and (now - last_kv2_ts) <= STALE_S
           
            if not (have_az and have_kv2):
                # Single-camera fallback if Azure only
                x = y = z = 0.0
                if have_az and K0 is not None and R0 is not None and t0 is not None:
                    try:
                        uv = np.array(last_az_pts[0], dtype=np.float64)
                        C, dir_w = _ray_from_pixel(K0, R0, t0, uv)
                        Xw = _intersect_ray_plane(C, dir_w, plane)
                        if Xw is not None and np.all(np.isfinite(Xw)):
                            x, y, z = float(Xw[0]), float(Xw[1]), float(Xw[2])
                            # optional debug
                            # print(f"[Fallback] {x:.3f} {y:.3f} {z:.3f}")
                    except Exception as e:
                        print(f"[Fallback] Ray-plane failed: {e}")
                else:
                    # detailed diagnostics
                    age_az = (now - last_az_ts) if last_az_ts else None
                    age_kv2 = (now - last_kv2_ts) if last_kv2_ts else None

                    if age_az is None:
                        print("[Fuse] No Azure detection yet.")
                    elif age_az > STALE_S:
                        print(f"[Fuse] Azure detection stale: {age_az:.3f}s")
                    if age_kv2 is None:
                        print("[Fuse] No Kinect v2 detection yet.")
                    elif age_kv2 > STALE_S:
                        print(f"[Fuse] Kinect v2 detection stale: {age_kv2:.3f}s")
                if stream and conn:
                    try:
                        print(f"Sending fallback point: {x} {y} {z}")
                        conn.sendall(f"{x} {y} {z}\n".encode("utf-8"))
                    except Exception:
                        pass
                time.sleep(0.01)
                continue

            # build image points for triangulation (two dots)
            image_points = [last_az_pts, last_kv2_pts]
            
            try:
                image_points, _ = find_point_correspondance_and_object_points(
                    [last_az_pts, last_kv2_pts], camera_poses, 2
                )
                print("DEBUG: image_points after correspondance:", image_points)
            except Exception as e:
                print(f"[Fuse] Triangulation call failed: {e}")
                image_points = []

            best3d = None
           
            if _is_nonempty(image_points):
                best3d = _select_best_3d_point(image_points, cluster_threshold=0.06)
    
            if best3d is not None and np.all(np.isfinite(best3d)):
                x, y, z = float(best3d[0]), float(best3d[1]), float(best3d[2])
            else:
                x, y, z = 0.0, 0.9, 0.0

            if stream and conn:
                try:
                    # print(f"Sending fused point: {x} {y} {z}")
                    conn.sendall(f"{x} {y} {z}\n".encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    continue
            else:
                print(f"Fused point: {x} {y} {z}")

            time.sleep(0.005)
        except Exception as e:
            print(f"Fused tracking error: {e}")
            time.sleep(0.01)
            continue
    # cleanup
    if stream:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            if server:
                server.close()
        except Exception:
            pass

def main():
    global running
    try:
        print("MoCap fused tracking - Azure Kinect DK + Kinect v2")
        
        # initialize Azure library
        pykinect.initialize_libraries(track_body=False)

        #Unity camera feed thread
        main_camera_thread = threading.Thread(target=main_camera.main, daemon=True)
        main_camera_thread.start()

        # processing thread
        proc = threading.Thread(target=track, args=(data_queue_azure, data_queue_kv2, True))
        proc.daemon = True
        proc.start()

        # start Azure thread
        def azure_runner():
            try:
                cfg = pykinect.default_configuration
                cfg.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
                cfg.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
                cfg.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
                k = pykinect.start_device(config=cfg)
                track_points_azure(k, data_queue_azure, preview=True)
            except Exception as e:
                print(f"Azure init error: {e}")

        t_azure = threading.Thread(target=azure_runner, daemon=True)
        t_azure.start()

        # start Kinect v2 thread
        t_kv2 = None
        if PyKinectRuntime is not None:
            def kv2_runner():
                try:
                    k2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
                    track_points_kinect_v2(k2, data_queue_kv2, preview=True)
                    try:
                        k2.close()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Kinect v2 init error: {e}")

            t_kv2 = threading.Thread(target=kv2_runner, daemon=True)
            t_kv2.start()
        else:
            print("PyKinect2 not available; Kinect v2 thread not started.")

        # main loop waits for interrupt
        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            running.clear()

        # join threads
        t_azure.join(timeout=2)
        if t_kv2:
            t_kv2.join(timeout=2)
        proc.join(timeout=2)
        return True

    except Exception as e:
        print(f"Main error: {e}")
        return False


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f'Unexpected error: {e}')
        sys.exit(1)
