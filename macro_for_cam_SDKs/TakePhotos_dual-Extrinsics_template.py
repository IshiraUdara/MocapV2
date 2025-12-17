import sys
import time
import threading
import os
import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a, Image

# Optional Kinect v2 (Kinect for Xbox One) support
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
except Exception:
    PyKinectRuntime = None
    PyKinectV2 = None

running = threading.Event()
running.set()

# Per-camera photo events (0 = Azure, 1 = Kinect v2)
take_photo_events = {0: threading.Event(), 1: threading.Event()}
take_photo_events[0].clear()
take_photo_events[1].clear()

def get_camera_event(cam_num):
    if cam_num not in take_photo_events:
        take_photo_events[cam_num] = threading.Event()
        take_photo_events[cam_num].clear()
    return take_photo_events[cam_num]

def trigger_all_cameras():
    for ev in take_photo_events.values():
        ev.set()

# --- Azure Kinect DK IR handling ---
def acquire_and_display_images_azure(kinect, cam_num=0, flipped=True, floor=False):
    camera_event = get_camera_event(cam_num)
    ir_image = None
    ir_image_8bit = None

    try:
        device_serial = kinect.get_serialnum()
        window_name = f'Azure Kinect IR - {device_serial}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Create output dirs
        base_dir = f'macro_for_cam_SDKs/Kinect_Azure_x_Kinect_OLD'
        if floor:
            out_dir = os.path.join(base_dir, 'tracking', f'cam{cam_num}')
        else:
            out_dir = os.path.join(base_dir, 'captured_images', f'cam{cam_num}')
        os.makedirs(out_dir, exist_ok=True)

        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1.0)

        while running.is_set():
            try:
                capture = kinect.get_capture()
                if capture is None:
                    time.sleep(0.001)
                    continue

                ir_handle = _k4a.k4a_capture_get_ir_image(capture.handle if hasattr(capture, 'handle') else capture)
                if not ir_handle:
                    # no IR image in this capture
                    continue

                try:
                    w = _k4a.k4a_image_get_width_pixels(ir_handle)
                    h = _k4a.k4a_image_get_height_pixels(ir_handle)
                    size = _k4a.k4a_image_get_size(ir_handle)
                    if w <= 0 or h <= 0 or size <= 0:
                        _k4a.k4a_image_release(ir_handle)
                        continue

                    # convert to numpy
                    arr = Image(ir_handle).to_numpy()
                    # Image.to_numpy typically returns np.ndarray
                    ir_image = arr if not isinstance(arr, tuple) else arr[1]

                    if ir_image is None:
                        _k4a.k4a_image_release(ir_handle)
                        continue

                    # convert 16-bit to 8-bit for display
                    if ir_image.dtype == np.uint16:
                        ir_image_8bit = (ir_image >> 8).astype(np.uint8)
                    else:
                        ir_image_8bit = ir_image.astype(np.uint8)

                    # to BGR for display
                    if ir_image_8bit.ndim == 2:
                        display = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)
                    else:
                        display = ir_image_8bit

                    if flipped:
                        display = cv2.flip(display, 1)

                finally:
                    _k4a.k4a_image_release(ir_handle)

                # FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / max((end_time - start_time), 1e-6)
                    frame_count = 0
                    start_time = end_time

                cv2.putText(display, f"Azure IR FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running.clear()
                    break
                elif key == ord('s'):
                    trigger_all_cameras()

                # Save if requested for this camera
                if camera_event.is_set() and ir_image_8bit is not None:
                    ts = int(time.time())
                    out_path = os.path.join(out_dir, f'{ts}.png')
                    # resize/save as 3-channel BGR for compatibility
                    resized = cv2.resize(ir_image_8bit, (1280, 720), interpolation=cv2.INTER_AREA)
                    if resized.ndim == 2:
                        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = resized
                    cv2.imwrite(out_path, bgr)
                    camera_event.clear()

                # small sleep to yield
                time.sleep(0.001)

            except Exception as e:
                # continue on per-frame errors
                print(f"Azure loop error: {e}")
                time.sleep(0.01)
                continue

        cv2.destroyWindow(window_name)
        return True

    except Exception as e:
        print(f'Azure acquisition error: {e}')
        return False

def run_single_camera_azure(device_id=0, cam_num=0, flipped=True, floor=False):
    try:
        pykinect.initialize_libraries(track_body=False)
        cfg = pykinect.default_configuration
        cfg.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        cfg.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        cfg.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        kinect = pykinect.start_device(config=cfg)
        return acquire_and_display_images_azure(kinect, cam_num, flipped, floor)
    except Exception as e:
        print(f'Failed to start Azure device {device_id}: {e}')
        return False

# --- Kinect v2 (Kinect for Xbox One) IR handling ---
def acquire_and_display_images_kinect_v2(kinect_runtime, cam_num=1, flipped=True, floor=False):
    if PyKinectRuntime is None or PyKinectV2 is None:
        print("PyKinect2 not available. Skipping Kinect v2.")
        return False

    camera_event = get_camera_event(cam_num)

    try:
        window_name = f'Kinect v2 IR - cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        base_dir = f'macro_for_cam_SDKs/Kinect_Azure_x_Kinect_OLD'
        if floor:
            out_dir = os.path.join(base_dir, 'tracking', f'cam{cam_num}')
        else:
            out_dir = os.path.join(base_dir, 'captured_images', f'cam{cam_num}')
        os.makedirs(out_dir, exist_ok=True)

        # try to get IR size; fallback to known Kinect v2 size
        try:
            ir_desc = kinect_runtime.infrared_frame_desc
            h, w = ir_desc.Height, ir_desc.Width
        except Exception:
            h, w = 424, 512

        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1.0)

        while running.is_set():
            try:
                ir_frame = kinect_runtime.get_last_infrared_frame()
                if ir_frame is None:
                    time.sleep(0.001)
                    continue

                try:
                    ir_image = ir_frame.reshape((h, w)).astype(np.uint16)
                except Exception:
                    ir_image = np.array(ir_frame, dtype=np.uint16).reshape((h, w))

                ir_image_8bit = (ir_image >> 8).astype(np.uint8)
                display = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)
                if flipped:
                    display = cv2.flip(display, 1)

                # FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / max((end_time - start_time), 1e-6)
                    frame_count = 0
                    start_time = end_time

                cv2.putText(display, f"Kinect v2 IR FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running.clear()
                    break
                elif key == ord('s'):
                    trigger_all_cameras()

                if camera_event.is_set():
                    ts = int(time.time())
                    out_path = os.path.join(out_dir, f'{ts}.png')
                    resized = cv2.resize(ir_image_8bit, (1280, 720), interpolation=cv2.INTER_AREA)
                    bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(out_path, bgr)
                    camera_event.clear()

                time.sleep(0.001)

            except Exception as e:
                print(f'Kinect v2 loop error: {e}')
                time.sleep(0.01)
                continue

        cv2.destroyWindow(window_name)
        return True

    except Exception as e:
        print(f'Kinect v2 acquisition error: {e}')
        return False

def run_single_camera_kinect_v2(device_id=0, cam_num=1, flipped=True, floor=False):
    if PyKinectRuntime is None or PyKinectV2 is None:
        print("PyKinect2 not installed; skipping Kinect v2 thread.")
        return False
    try:
        kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
        try:
            return acquire_and_display_images_kinect_v2(kinect, cam_num, flipped, floor)
        finally:
            try:
                kinect.close()
            except Exception:
                pass
    except Exception as e:
        print(f'Failed to start Kinect v2 runtime: {e}')
        return False

# --- Combined launcher (Azure + Kinect v2) ---
def main_combined(auto=False, floor=False, flipped=True):
    global running
    try:
        print('Starting: Azure Kinect DK (cam0) + Kinect v2 (cam1)')
        # Start Azure thread
        t_azure = threading.Thread(target=run_single_camera_azure, args=(0, 0, flipped, floor))
        t_azure.daemon = True
        t_azure.start()

        # Start Kinect v2 thread if available
        t_v2 = None
        if PyKinectRuntime is not None:
            t_v2 = threading.Thread(target=run_single_camera_kinect_v2, args=(0, 1, flipped, floor))
            t_v2.daemon = True
            t_v2.start()
        else:
            print("PyKinect2 not available; Kinect v2 thread not started.")

        time.sleep(6.0)

        # main loop: support auto photo mode
        while running.is_set():
            time.sleep(0.5)
            if auto:
                trigger_all_cameras()
                time.sleep(0.5)

        # join threads
        t_azure.join(timeout=2)
        if t_v2:
            t_v2.join(timeout=2)

        print('Done.')
        return True

    except Exception as e:
        print(f'Main error: {e}')
        return False

if __name__ == '__main__':
    try:
        success = main_combined(auto=False, floor=True, flipped=True)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f'Unexpected: {e}')
        sys.exit(1)
