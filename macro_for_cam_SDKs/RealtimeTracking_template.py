import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import *
from pykinect_azure.k4a import _k4a
from lib.Helpers import find_point_correspondance_and_object_points, get_extrinsics
import queue
import socket
import msgpack

import numpy as np
import mmap

# Try to import with CUDA, fallback to CPU if needed
use_cuda = False

try:
    from lib.ImageOperations import _find_dot
    use_cuda = True
    print("Using CUDA-accelerated dot detection")
except Exception as cuda_error:
    print(f"CUDA error: {cuda_error}")
    print("Falling back to CPU-only mode...")
    use_cuda = False

running = threading.Event()
running.set()

camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")

def track_points_Azure(kinect, data_queue: queue.Queue, preview=False):
    """
    Continuously acquires IR images from Azure Kinect DK and processes them.
    """
    global running
    try:
        device_serial_number = kinect.get_serialnum()
        print(f'Device serial number: {device_serial_number}')
        
        window_name = f'Kinect IR - {device_serial_number}'
        if preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print('Starting IR image acquisition...')
        
        frame_count = 0
        start_time = time.time()
        time.sleep(1)  # Allow camera to stabilize
        
        while running.is_set():
            try:
                capture = kinect.get_capture()
                
                if capture is not None:
                    # Get IR image handle
                    ir_image_handle = _k4a.k4a_capture_get_ir_image(capture)
                    if ir_image_handle:
                        try:
                            width = _k4a.k4a_image_get_width_pixels(ir_image_handle)
                            height = _k4a.k4a_image_get_height_pixels(ir_image_handle)
                            buffer_size = _k4a.k4a_image_get_size(ir_image_handle)
                            
                            if width > 0 and height > 0 and buffer_size > 0:
                                ir_result = Image(ir_image_handle).to_numpy()
                                if isinstance(ir_result, tuple):
                                    success, ir_image = ir_result
                                    if not success or ir_image is None:
                                        print("Failed to convert IR image to numpy array")
                                        _k4a.k4a_image_release(ir_image_handle)
                                        continue
                                else:
                                    ir_image = ir_result
                                
                                # Convert 16-bit IR to 8-bit for processing/display
                                if ir_image.dtype == np.uint16:
                                    gray_image = (ir_image / 256).astype(np.uint8)
                                else:
                                    # already 8-bit
                                    gray_image = ir_image.astype(np.uint8)
                            else:
                                print(f"Invalid IR image dimensions: {width}x{height}, buffer size: {buffer_size}")
                                _k4a.k4a_image_release(ir_image_handle)
                                continue
                        except Exception as img_ex:
                            print(f"Error converting IR image: {img_ex}")
                            _k4a.k4a_image_release(ir_image_handle)
                            continue
                        finally:
                            _k4a.k4a_image_release(ir_image_handle)
                    else:
                        # no IR image in this capture
                        continue
                    
                    # Process grayscale IR image for dot detection
                    try:
                        processed_image, detected_points = _find_dot(gray_image, print_location=True)
                    except Exception as proc_ex:
                        print(f"Dot detection error: {proc_ex}")
                        continue

                    try:
                        if data_queue.full():
                            data_queue.get_nowait()
                        data_queue.put_nowait(detected_points)
                    except queue.Full:
                        print("Queue is full")
                    
                    frame_count += 1
                    
                    if preview:
                        # processed_image expected to be single-channel or BGR; ensure displayable
                        if len(processed_image.shape) == 2:
                            disp = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
                        else:
                            disp = processed_image
                        cv2.imshow(window_name, disp)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            running.clear()
                            break
                    
            except Exception as ex:
                print(f'Error during IR capture: {ex}')
                time.sleep(0.01)  # Small delay to prevent busy waiting

        if preview:
            cv2.destroyWindow(window_name)
        print("Kinect IR feed stopped")
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False
        
    return True

def run_single_camera_Azure(device_id, data_queue):
    """
    Initialize and run a single Kinect device.
    """
    try:
        pykinect.initialize_libraries(track_body=False)
        
        device_config = pykinect.default_configuration
        #device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        kinect = pykinect.start_device(config=device_config)
        print(f'Kinect {device_id} initialized successfully')
        
        result = track_points_Azure(kinect, data_queue, preview=True)
        
        return result
        
    except Exception as ex:
        print(f'Error initializing Kinect {device_id}: {ex}')
        return False

# ⚠️ Update with your correct RTSP URL
# Try variations if needed:
# rtsp://admin:PASSWORD@192.168.x.x:554/h264
# rtsp://admin:PASSWORD@192.168.x.x:554/Streaming/Channels/101
RTSP_URL = "rtsp://admin:WOJWUD@169.254.27.194:554/h264"


def track_points_EZVIZ(rtsp_url, data_queue: queue.Queue, preview=False):
    """
    Continuously acquires images from EZVIZ H3C via RTSP and processes them.
    """
    global running
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("❌ Failed to open EZVIZ RTSP stream")
            return False

        window_name = f"EZVIZ H3C Feed"
        if preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Starting EZVIZ image acquisition...")

        frame_count = 0
        start_time = time.time()
        time.sleep(1)  # Allow camera to stabilize

        while running.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ No frame received from EZVIZ camera")
                time.sleep(0.05)
                continue

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            processed_image, detected_points = _find_dot(gray_image, print_location=True)

            try:
                if data_queue.full():
                    data_queue.get_nowait()
                data_queue.put_nowait(detected_points)
            except queue.Full:
                print("Queue is full")

            frame_count += 1

            if preview:
                cv2.imshow(window_name, processed_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running.clear()
                    break

        cap.release()
        if preview:
            cv2.destroyWindow(window_name)
        print("EZVIZ feed stopped")

    except Exception as ex:
        print(f"Error: {ex}")
        return False

    return True


def track(data_queue_Azure: queue.Queue, data_queue_EZVIZ: queue.Queue, stream=True):
    """
    Process points from 1 or 2 EZVIZ cameras and send to Unity if enabled.
    """
    global camera_poses
    print(camera_poses)
    print("Tracking started")

    if stream:
        HOST = "127.0.0.1"
        PORT = 5002
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(1)
        print("Waiting for Unity to connect...")
        connection, _ = server.accept()
        print("Connected!")

    point = [0, 0, 0, 0, 0, 0, 0, 0]
    fps = 0
    old_time = time.time()

    while running.is_set():
        fps = time.time() - old_time
        old_time = time.time()
        fps = 1 / fps if fps > 0 else 0

        try:
            if not (data_queue_Azure.empty() or data_queue_EZVIZ.empty()):
                data1 = data_queue_Azure.get_nowait()
                data2 = data_queue_EZVIZ.get_nowait()
                image_points = [data1, data2]
                object_points, image_p = find_point_correspondance_and_object_points(
                    image_points, camera_poses, 4
                )

                if stream:
                    if len(object_points) > 0:
                        point = object_points[0]
                        point = list(point)
                        point = [0, 0, 0, 0] + point
                    data = {"tracker1": point}
                    try:
                        connection.send(msgpack.packb(data, use_bin_type=True))
                        print(f"Object Points: {point}")
                    except (ConnectionResetError, BrokenPipeError):
                        print("\nUnity disconnected, waiting for reconnection...")
                        connection, _ = server.accept()
                        print("Connected!")
                        continue
                else:
                    print(f"Object Points: {object_points}")
                print(f"Image Points: {image_p}")
                print(f"FPS: {fps:.2f}")

        except queue.Empty:
            pass
        except Exception as ex:
            print(f"Tracking error: {ex}")

        time.sleep(0.01)


def run_single_camera_EZVIZ(rtsp_url, data_queue):
    """
    Initialize and run a single EZVIZ H3C camera.
    """
    try:
        print(f"Initializing EZVIZ camera: {rtsp_url}")
        result = track_points_EZVIZ(rtsp_url, data_queue, preview=True)
        return result
    except Exception as ex:
        print(f"Error initializing EZVIZ camera: {ex}")
        return False


def main():
    """
    Main entry point.
    """
    global running
    try:
        print("MoCap v2.0 - EZVIZ H3C")

        pykinect.initialize_libraries(track_body=False)
        print(f'MoCap v2.0 - Azure Kinect DK')

        data_queue_Azure = queue.Queue(maxsize=10)
        data_queue_EZVIZ = queue.Queue(maxsize=10)  # If you add a 2nd EZVIZ camera

        process_thread = threading.Thread(target=track, args=(data_queue_Azure, data_queue_EZVIZ))
        process_thread.daemon = True
        process_thread.start()

        EZVIZ_camera_thread = threading.Thread(target=run_single_camera_EZVIZ, args=(RTSP_URL, data_queue_EZVIZ))
        EZVIZ_camera_thread.daemon = True
        EZVIZ_camera_thread.start()

        Azure_camera_thread = threading.Thread(target=run_single_camera_Azure, args=(0, data_queue_Azure))
        Azure_camera_thread.daemon = True
        Azure_camera_thread.start()

        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            running.clear()

        print("Stopping cameras...")
        EZVIZ_camera_thread.join(timeout=2)
        Azure_camera_thread.join(timeout=2)

        print("\nDone!")
        return True

    except Exception as ex:
        print(f"Error: {ex}")
        return False

if __name__ == '__main__':
    try:
        success = main()
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
