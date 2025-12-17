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
    """Acquire IR from Azure Kinect DK, detect dots and push detections (list of [x,y])."""
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

                    gray = (ir_image >> 8).astype(np.uint8) if ir_image.dtype == np.uint16 else ir_image.astype(np.uint8)

                finally:
                    _k4a.k4a_image_release(ir_handle)

                processed, detected = _find_dot(gray, print_location=False)
                # push latest
                try:
                    if out_queue.full():
                        out_queue.get_nowait()
                    out_queue.put_nowait(detected)
                except queue.Full:
                    pass

                if preview:
                    disp = processed if processed.ndim == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    cv2.imshow(win, disp)
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

                gray = (ir_image >> 8).astype(np.uint8)

                processed, detected = _find_dot(gray, print_location=False)
                try:
                    if out_queue.full():
                        out_queue.get_nowait()
                    out_queue.put_nowait(detected)
                except queue.Full:
                    pass

                if preview:
                    disp = processed if processed.ndim == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    cv2.imshow(win, disp)
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


def track(out_queue_azure: queue.Queue, out_queue_kv2: queue.Queue, stream=True):
    """Fuse detections from both cameras, triangulate and send best single 3D point.
       Server-only mode: Python acts as TCP server; Unity should connect as client.

       Communication matches test.py style: plain UTF-8 lines sent over TCP.
       Each line: "x y z" (space separated, meters), newline-delimited.
    """
    global camera_poses
    print("Fused tracking started (server mode)")

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

            # get newest detections
            azure_pts = _drain_latest(out_queue_azure)
            kv2_pts = _drain_latest(out_queue_kv2)

            if azure_pts is None and kv2_pts is None:
                time.sleep(0.005)
                continue

            image_points = []
            image_points.append(azure_pts if azure_pts is not None else [])
            image_points.append(kv2_pts if kv2_pts is not None else [])
        
            # triangulate / find correspondences
            object_points, image_p = find_point_correspondance_and_object_points(image_points, camera_poses, 1)
          
            best3d = None
            if _is_nonempty(object_points):
                best3d = _select_best_3d_point(object_points, cluster_threshold=0.06)

            if best3d is not None and np.all(np.isfinite(best3d)):
                x, y, z = float(best3d[0]), float(best3d[1]), float(best3d[2])
                print(f"Fused point: {x:.3f} {y:.3f} {z:.3f}")
            else:
                x, y, z = 0.0, 0.0, 0.0

            x_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
            y_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
            z_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
            if stream and conn:
                try:
                    x_random = random.choice(x_vals)
                    y_random = random.choice(y_vals)
                    z_random = random.choice(z_vals)
                    # plain UTF-8 line: "x y z\n"
                    #time.sleep(1)  # simulate processing 
                   
                    time.sleep(0.1)  # simulate ~60Hz
                    line = f"{x} {y} {z}\n".encode("utf-8")
                    conn.sendall(line)
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    print("Unity disconnected or send failed:", repr(e))
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    continue
            else:
                print(f"Fused point: {x} {y} {z}")

            # small sleep to limit CPU
            time.sleep(0.005)

        except Exception as e:
            print(f"Tracking error: {e}")
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
