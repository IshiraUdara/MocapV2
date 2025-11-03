import sys
import time
import threading
import cv2
import os
import numpy as np

# PyKinect v2 (Kinect for Xbox One / SDK v2.0_1409)
from pykinect2 import PyKinectRuntime, PyKinectV2

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

def acquire_and_display_images(kinect: PyKinectRuntime.PyKinectRuntime, cam_num, flipped=True, floor=False):
    """
    Acquire IR frames from Kinect v2 and display/save images.
    Uses pykinect2 PyKinectRuntime with FrameSourceTypes_Infrared.
    """
    global running, take_photo
    try:
        window_name = f'Kinect v2 IR Feed - cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print('Starting Kinect v2 IR acquisition...')
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)

        # Prepare output directories
        if floor:
            os.makedirs(f'Kinect_DK_cam/tracking/cam{cam_num}', exist_ok=True)
        else:
            os.makedirs(f'Kinect_DK_cam/captured_images/cam{cam_num}', exist_ok=True)

        # Determine infrared frame dimensions (fallback to known Kinect v2 IR size)
        try:
            ir_desc = kinect.infrared_frame_desc
            height, width = ir_desc.Height, ir_desc.Width
        except Exception:
            height, width = 424, 512

        while running.is_set():
            try:
                ir_frame = kinect.get_last_infrared_frame()
                if ir_frame is None:
                    time.sleep(0.001)
                    continue

                # ir_frame is a 1D numpy array (uint16), reshape to HxW
                try:
                    ir_image = ir_frame.reshape((height, width)).astype(np.uint16)
                except Exception:
                    # fallback if shape differs
                    ir_image = ir_frame.copy().astype(np.uint16)
                    ir_image = ir_image.reshape((height, width))

                # Convert 16-bit IR to 8-bit for display
                ir_image_8bit = np.right_shift(ir_image, 8).astype(np.uint8)

                # Convert grayscale to BGR for GUI
                image_data = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)

                # Optional horizontal flip
                if flipped:
                    image_data_flipped = cv2.flip(image_data, 1)
                else:
                    image_data_flipped = image_data

                # FPS calc
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time + 1e-6)
                    frame_count = 0
                    start_time = end_time

                cv2.putText(image_data_flipped, f"IR FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow(window_name, image_data_flipped)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print('ESC pressed. Exiting...')
                    running.clear()
                    break
                elif key == ord('s'):  # save on 's'
                    take_photo.set()

                if take_photo.is_set():
                    ts = int(time.time())
                    if floor:
                        out_path = f'Kinect_DK_cam/tracking/cam{cam_num}/{ts}_ir.png'
                    else:
                        out_path = f'Kinect_DK_cam/captured_images/cam{cam_num}/{ts}_ir.png'
                    # Save 8-bit PNG for compatibility; optionally save 16-bit by cv2.imwrite with ir_image (uint16)
                    cv2.imwrite(out_path, ir_image_8bit)
                    print(f"Saved IR image: {out_path}")
                    take_photo.clear()

                time.sleep(0.001)

            except Exception as ex:
                print(f'Error during IR capture loop: {ex}')
                time.sleep(0.01)
                continue

        cv2.destroyWindow(window_name)
        print("Kinect v2 IR feed stopped")

    except Exception as ex:
        print(f'Error: {ex}')
        return False

    return True


def run_single_camera(device_id=0, cam_num=0, flipped=True, floor=True):
    """
    Initialize Kinect v2 runtime and run IR acquisition.
    Note: Kinect v2 supports a single device per machine in typical setups.
    """
    try:
        # initialize Kinect runtime for infrared frames
        kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
        print('Kinect v2 initialized for IR capture')

        result = acquire_and_display_images(kinect, cam_num, flipped, floor)

        # cleanup runtime
        kinect.close()
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