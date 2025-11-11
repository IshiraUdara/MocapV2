import sys
import time
import threading
import cv2
import os
import numpy as np
import ctypes
import types

# PyKinect v2 (Kinect for Xbox One / SDK v2.0_1409)
from pykinect2 import PyKinectRuntime, PyKinectV2

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

def acquire_and_display_images(kinect: PyKinectRuntime.PyKinectRuntime, cam_num, flipped=True, floor=False):
    """
    Robust IR acquisition with a watchdog:
     - prefers IR getter, falls back to depth
     - warmup wait to let sensor settle
     - if no valid frames for `max_no_frame_count` iterations -> close window and exit
    """
    global running, take_photo
    try:
        window_name = f'Kinect v2 IR Feed - cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # determine frame size
        try:
            ir_desc = getattr(kinect, 'infrared_frame_desc', None)
            if ir_desc:
                height, width = ir_desc.Height, ir_desc.Width
            else:
                depth_desc = getattr(kinect, 'depth_frame_desc', None)
                if depth_desc:
                    height, width = depth_desc.Height, depth_desc.Width
                else:
                    height, width = 424, 512
        except Exception:
            height, width = 424, 512

        # select getter
        getter = None
        getter_name = None
        if hasattr(kinect, 'get_last_infrared_frame'):
            getter = lambda: kinect.get_last_infrared_frame()
            getter_name = 'get_last_infrared_frame'
        elif hasattr(kinect, 'get_last_depth_frame'):
            getter = lambda: kinect.get_last_depth_frame()
            getter_name = 'get_last_depth_frame (fallback)'
        else:
            getter = None
            getter_name = 'none'

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
                        a0 = np.array(raw0, dtype=np.uint16)
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

        # watchdog: if we get too many consecutive "no frame" results, quit gracefully
        # increased threshold to allow slow sensors to start
        max_no_frame_count = int(800)   # ~800 * 0.01s = ~8s
        no_frame_count = 0

        while running.is_set():
            try:
                if getter is None:
                    print("No IR/depth getter available on this PyKinectRuntime instance.")
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

                # raw is expected as 1D uint16 array-like
                try:
                    arr = np.array(raw, dtype=np.uint16)
                    if arr.size == 0:
                        no_frame_count += 1
                        time.sleep(0.01)
                        continue
                    ir_image = arr.reshape((height, width))
                except Exception:
                    if isinstance(raw, np.ndarray) and raw.ndim == 2:
                        ir_image = raw.astype(np.uint16)
                    else:
                        try:
                            ir_image = np.array(raw, dtype=np.uint16).reshape((height, width))
                        except Exception:
                            print("Failed to reshape incoming frame, skipping.")
                            time.sleep(0.01)
                            continue

                # Convert 16-bit IR to 8-bit for display.
                # Use dynamic normalization (min/max) because raw IR may be near-zero or high-range.
                vmin = int(ir_image.min())
                vmax = int(ir_image.max())
                if vmax <= vmin:
                    # flat image -> display as zeros (avoid division by zero)
                    ir8 = np.zeros((height, width), dtype=np.uint8)
                else:
                    # normalize to full 0-255 range for visibility
                    ir8 = ((ir_image.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))).clip(0, 255).astype(np.uint8)
                # optional local contrast boost (uncomment if needed):
                # ir8 = cv2.equalizeHist(ir8)
                disp = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)

                # occasional debug overlay so you can see raw range
                if frame_count % 30 == 0:
                    print(f"IR stats: min={vmin} max={vmax} mean={int(ir_image.mean())}")
                cv2.putText(disp, f"min:{vmin} max:{vmax}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if flipped:
                    disp = cv2.flip(disp, 1)

                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / max((end_time - start_time), 1e-6)
                    frame_count = 0
                    start_time = end_time

                cv2.putText(disp, f"IR FPS: {fps:.1f} [{getter_name}]", (10, 30),
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
                    out_dir = f'Kinect_DK_cam/tracking/cam{cam_num}' if floor else f'Kinect_DK_cam/captured_images/cam{cam_num}'
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'{ts}_ir.png')
                    cv2.imwrite(out_path, ir8)
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
        print(f"IR acquisition error: {ex}")
        return False


def run_single_camera(device_id=0, cam_num=0, flipped=True, floor=True):
    """
    Initialize Kinect v2 runtime and run IR acquisition.
    Note: Kinect v2 supports a single device per machine in typical setups.
    """
    try:
        # initialize Kinect runtime for infrared + depth frames (give runtime both sources)
        src = PyKinectV2.FrameSourceTypes_Infrared | PyKinectV2.FrameSourceTypes_Depth
        kinect = PyKinectRuntime.PyKinectRuntime(src)
        print('Kinect v2 initialized for IR capture')

        # small extra wait after init to let runtime open device
        time.sleep(1.5)
        
        result = acquire_and_display_images(kinect, cam_num, flipped, floor)

        # cleanup runtime
        try:
            kinect.close()
        except Exception:
            pass
        return result

    except Exception as ex:
        print(f'Error initializing Kinect v2: {ex}')
        return False


def main(auto=False, floor=False, flipped=True):
    """
    Main entry: start a single Kinect v2 thread for IR capture.
    """
    global running
    try:
        print('Kinect v2 IR Capture (SDK v2.0_1409)')

        # Start single camera thread (Kinect v2 typically only one)
        t = threading.Thread(target=run_single_camera, args=(0, 0, flipped, floor))
        t.daemon = True
        t.start()

        time.sleep(2)  # allow init

        while running.is_set():
            time.sleep(0.5)
            if auto:
                take_photo.set()
                time.sleep(0.5)
                take_photo.clear()

        t.join(timeout=2)
        print('Stopping IR capture')
        return True

    except Exception as ex:
        print(f'Error: {ex}')
        return False


if __name__ == '__main__':
    try:
        success = main(auto=False, floor=False, flipped=True)
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)