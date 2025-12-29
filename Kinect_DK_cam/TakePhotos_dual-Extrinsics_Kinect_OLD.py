import sys
import time
import threading
import cv2
import os
import numpy as np
import ctypes
import types
import json

# PyKinect v2 (Kinect for Xbox One / SDK v2.0_1409)
from pykinect2 import PyKinectRuntime, PyKinectV2
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a, Image

# --- monkeypatch class to provide get_last_infrared_frame if missing ---
def _pykinect_get_last_infrared_frame(self):
    """
    Return a 1D uint16 numpy array for the last infrared frame if available.
    Prefer internal _last_infrared_frame buffer; fallback to get_last_depth_frame().
    """
    try:
        # prefer explicitly stored last infrared buffer
        ir = getattr(self, "_last_infrared_frame", None)
        if ir is not None:
            return np.copy(ir) if isinstance(ir, np.ndarray) else np.array(ir, dtype=np.uint16)
        # some builds store _infrared_frame_data as ctypes buffer
        buf = getattr(self, "_infrared_frame_data", None)
        cap = getattr(self, "_infrared_frame_data_capacity", None)
        if buf is not None and cap is not None:
            try:
                # try to convert ctypes buffer to numpy
                arr = np.ctypeslib.as_array(buf, shape=(int(cap.value),))
                return np.copy(arr.astype(np.uint16))
            except Exception:
                pass
        # fallback to last depth frame (same resolution on Kinect v2)
        if hasattr(self, "get_last_depth_frame"):
            try:
                return self.get_last_depth_frame()
            except Exception:
                pass
    except Exception:
        pass
    return None

# Patch class before creating any instances
if not hasattr(PyKinectRuntime.PyKinectRuntime, "get_last_infrared_frame"):
    setattr(PyKinectRuntime.PyKinectRuntime, "get_last_infrared_frame", _pykinect_get_last_infrared_frame)

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

latest_kv2_color = {"frame": None, "size": (0, 0)}
latest_azure_color = {"frame": None}

# Load intrinsics once (adjust path if your JSON lives elsewhere)
INTRINSICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'macro_for_cam_SDKs', 'jsons', 'camera-params-in.json')
try:
    with open(INTRINSICS_PATH, 'r') as f:
        CAPARAMS = json.load(f)
except Exception:
    CAPARAMS = None

def _get_cam_intrinsics(cam_idx):
    if CAPARAMS is None:
        return None
    try:
        key = f"cam{cam_idx}"
        cam = CAPARAMS.get(key)
        if cam is None and isinstance(CAPARAMS, list) and cam_idx < len(CAPARAMS):
            cam = CAPARAMS[cam_idx]
        if cam is None:
            return None
        K = np.array(cam.get("K") or cam.get("camera_matrix") or cam.get("intrinsics"))
        dist = np.array(cam.get("distCoeffs") or cam.get("distortion_coefficients") or cam.get("dist"))
        if K.size and dist.size:
            return K.astype(np.float32), dist.astype(np.float32)
    except Exception:
        pass
    return None

def _undistort_color(disp_bgr, cam_idx):
    intr = _get_cam_intrinsics(cam_idx)
    if intr is None:
        return disp_bgr
    K, dist = intr
    h, w = disp_bgr.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0)
    und = cv2.undistort(disp_bgr, K, dist, None, newK)
    x, y, ww, hh = roi
    if ww > 0 and hh > 0:
        und = und[y:y+hh, x:x+ww]
    return und

def acquire_and_display_color_kv2(kinect: PyKinectRuntime.PyKinectRuntime, cam_num, flipped=True, floor=False):
    global running, take_photo, latest_kv2_color
    try:
        window_name = f'Kinect v2 Color - cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            cdesc = getattr(kinect, 'color_frame_desc', None)
            if cdesc:
                height, width = cdesc.Height, cdesc.Width
            else:
                height, width = 1080, 1920
        except Exception:
            height, width = 1080, 1920

        getter = getattr(kinect, 'get_last_color_frame', None)
        getter_name = 'get_last_color_frame' if getter else 'none'
        print(f"[Kinect v2] selected frame getter: {getter_name}")

        # allow sensor to settle
        sensor_settle_seconds = 3.0
        print(f"Waiting {sensor_settle_seconds:.1f}s for sensor to settle...")
        time.sleep(sensor_settle_seconds)

        # warmup: wait until we receive at least one non-empty frame (bounded)
        warmup_seconds = 8.0
        warmup_poll = 0.05
        warmup_deadline = time.time() + warmup_seconds
        got_warmup = False
        if getter is not None:
            print(f"Warmup: polling for first frame for up to {warmup_seconds}s...")
            while time.time() < warmup_deadline and running.is_set():
                try:
                    raw0 = getter()
                except Exception:
                    raw0 = None
                if raw0 is not None:
                    try:
                        a0 = np.array(raw0)
                        if a0.size >= 16:
                            got_warmup = True
                            print("Warmup: first valid frame received.")
                            break
                    except Exception:
                        pass
                time.sleep(warmup_poll)
        if not got_warmup:
            print(f"Warmup timeout ({warmup_seconds}s). Continuing but frames may not arrive. Check drivers / SDK if no frames appear.")

        frame_count = 0
        start_time = time.time()
        fps = 0

        # watchdog
        max_no_frame_count = int(800)
        no_frame_count = 0

        while running.is_set():
            try:
                if getter is None:
                    print("No color getter available on this PyKinectRuntime instance.")
                    time.sleep(0.2)
                    break

                raw = getter()
                if raw is None:
                    no_frame_count += 1
                    # print diagnostic only occasionally
                    if no_frame_count % 100 == 0:
                        print(f"No frames yet (no_frame_count={no_frame_count}) using {getter_name}")
                    if no_frame_count >= max_no_frame_count:
                        print("No frames received for an extended period — closing window and exiting. Check Kinect SDK/drivers and comtypes wrapper.")
                        try:
                            cv2.destroyWindow(window_name)
                        except Exception:
                            pass
                        return False
                    time.sleep(0.01)
                    continue

                # reset watchdog on valid raw
                no_frame_count = 0

                # raw is expected as 1D uint8 BGRA array-like
                try:
                    arr = np.array(raw, dtype=np.uint8)
                    if arr.size == 0:
                        no_frame_count += 1
                        time.sleep(0.01)
                        continue
                    color_bgra = arr.reshape((height, width, 4))
                except Exception:
                    try:
                        color_bgra = raw.reshape((height, width, 4)).astype(np.uint8)
                    except Exception:
                        print("Failed to reshape incoming color frame, skipping.")
                        time.sleep(0.01)
                        continue

                disp = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)

                if flipped:
                    disp = cv2.flip(disp, 1)

                disp = _undistort_color(disp, cam_num)
                latest_kv2_color["frame"] = disp
                latest_kv2_color["size"] = (width, height)

                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / max((end_time - start_time), 1e-6)
                    frame_count = 0
                    start_time = end_time

                cv2.putText(disp, f"Color FPS: {fps:.1f} [{getter_name}]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow(window_name, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running.clear()
                    break
                elif key == ord('s'):
                    take_photo.set()

                if take_photo.is_set():
                    ts = int(time.time())
                    out_dir = f'Kinect_DK_cam/captured_images/cam{cam_num}'
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'{ts}_color_kv2.png')
                    cv2.imwrite(out_path, disp)  # undistorted
                    print("Saved:", out_path)
                    take_photo.clear()

                time.sleep(0.001)

            except Exception as ex:
                print(f"Error during IR capture loop: {ex}")
                time.sleep(0.01)
                continue

        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        return True

    except Exception as ex:
        print(f"Kinect v2 color acquisition error: {ex}")
        return False

def run_kv2_color(cam_num=1, flipped=True):
    try:
        src = PyKinectV2.FrameSourceTypes_Color
        kinect = PyKinectRuntime.PyKinectRuntime(src)
        print('Kinect v2 initialized for Color')
        time.sleep(1.5)
        result = acquire_and_display_color_kv2(kinect, cam_num, flipped, floor=False)
        try:
            kinect.close()
        except Exception:
            pass
        return result

    except Exception as ex:
        print(f'Error initializing Kinect v2: {ex}')
        return False


def acquire_and_display_color_azure(device, cam_num, flipped=True):
    global running, take_photo, latest_azure_color
    try:
        window_name = f'Azure Kinect Color - cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        time.sleep(1.0)
        while running.is_set():
            try:
                capture = device.get_capture()
                if capture is None:
                    time.sleep(0.001)
                    continue

                color_handle = _k4a.k4a_capture_get_color_image(capture.handle if hasattr(capture, 'handle') else capture)
                if not color_handle:
                    continue

                try:
                    w = _k4a.k4a_image_get_width_pixels(color_handle)
                    h = _k4a.k4a_image_get_height_pixels(color_handle)
                    size = _k4a.k4a_image_get_size(color_handle)
                    if w <= 0 or h <= 0 or size <= 0:
                        continue

                    arr = Image(color_handle).to_numpy()
                    color_bgra = arr if not isinstance(arr, tuple) else arr[1]
                    if color_bgra is None:
                        continue

                    if color_bgra.ndim == 3 and color_bgra.shape[2] == 4:
                        disp = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
                    else:
                        # if format already BGR
                        disp = color_bgra.copy()
                finally:
                    _k4a.k4a_image_release(color_handle)

                if flipped:
                    disp = cv2.flip(disp, 1)

                disp = _undistort_color(disp, cam_num)
                latest_azure_color["frame"] = disp
                cv2.imshow(window_name, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running.clear()
                    break
                elif key == ord('s'):
                    take_photo.set()

                if take_photo.is_set():
                    ts = int(time.time())
                    out_dir = f'Kinect_DK_cam/captured_images/cam0'
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'{ts}_color_azure.png')
                    cv2.imwrite(out_path, disp)  # undistorted
                    print("Saved:", out_path)
                    take_photo.clear()

                time.sleep(0.001)

            except Exception as e:
                print(f"Azure color loop error: {e}")
                time.sleep(0.01)
                continue

        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        return True

    except Exception as ex:
        print(f"Azure color acquisition error: {ex}")
        return False


def main(flipped=True):
    global running
    try:
        print('Dual Color Capture: Azure Kinect + Kinect v2')

        pykinect.initialize_libraries(track_body=False)

        cfg = pykinect.default_configuration
        cfg.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        cfg.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        cfg.depth_mode = pykinect.K4A_DEPTH_MODE_OFF
        cfg.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        azure = pykinect.start_device(config=cfg)

        t_azure = threading.Thread(target=acquire_and_display_color_azure, args=(azure, 0, flipped))
        t_azure.daemon = True
        t_azure.start()

        t_kv2 = threading.Thread(target=run_kv2_color, args=(1, flipped))
        t_kv2.daemon = True
        t_kv2.start()

        while running.is_set():
            time.sleep(0.1)

        t_azure.join(timeout=2)
        t_kv2.join(timeout=2)
        print('Stopping color capture')
        return True

    except Exception as ex:
        print(f'Error: {ex}')
        return False


if __name__ == '__main__':
    try:
        success = main(flipped=True)
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)