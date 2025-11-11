import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import *
from lib.ImageOperations import _find_dot
from pykinect_azure.k4a import _k4a
from lib.Helpers import find_point_correspondance_and_object_points, get_extrinsics
import queue
import socket
import msgpack

running = threading.Event()
running.set()
camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")


def track_points(kinect, data_queue: queue.Queue, preview=False):
    """
    Continuously acquires images from Azure Kinect DK and processes them.
    """
    global running
    try:
        device_serial_number = kinect.get_serialnum()
        print(f'Device serial number: {device_serial_number}')
        
        window_name = f'Kinect Feed - {device_serial_number}'
        if preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print('Starting image acquisition...')
        
        frame_count = 0
        start_time = time.time()
        time.sleep(1)  # Allow camera to stabilize
        
        while running.is_set():
            try:
                capture = kinect.get_capture()
                
                if capture is not None:
                    # Use the public API method to get color image
                    color_image_handle = _k4a.k4a_capture_get_color_image(capture)
                    if color_image_handle:
                        try:
                            # Get image properties first
                            width = _k4a.k4a_image_get_width_pixels(color_image_handle)
                            height = _k4a.k4a_image_get_height_pixels(color_image_handle)
                            buffer_size = _k4a.k4a_image_get_size(color_image_handle)
                            
                            # Check if we have valid dimensions
                            if width > 0 and height > 0 and buffer_size > 0:
                                # Convert to numpy array using the Image class
                                color_image_result = Image(color_image_handle).to_numpy()

                                # Check if the result is a tuple (success, array)
                                if isinstance(color_image_result, tuple):
                                    success, color_image = color_image_result
                                    if not success or color_image is None:
                                        print("Failed to convert image to numpy array")
                                        _k4a.k4a_image_release(color_image_handle)
                                        continue
                                else:
                                    color_image = color_image_result
                                
                                # Check if the result is actually a numpy array
                                if isinstance(color_image, tuple):
                                    print(f"Warning: Image conversion returned tuple: {color_image}")
                                    _k4a.k4a_image_release(color_image_handle)
                                    continue
                                    
                            else:
                                print(f"Invalid image dimensions: {width}x{height}, buffer size: {buffer_size}")
                                _k4a.k4a_image_release(color_image_handle)
                                continue
                                
                        except Exception as img_ex:
                            print(f"Error converting image: {img_ex}")
                            _k4a.k4a_image_release(color_image_handle)
                            continue
                        finally:
                            _k4a.k4a_image_release(color_image_handle)
                    else:
                        continue
                    
                    if color_image is not None:
                        # Convert from BGRA to BGR if needed
                        if color_image.shape[2] == 4:  # BGRA
                            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                        else:  # Already BGR
                            color_image_bgr = color_image
                            
                        gray_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2GRAY)
                        
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
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                running.clear()
                                break
                    
            except Exception as ex:
                print(f'Error during capture: {ex}')
                time.sleep(0.01)  # Small delay to prevent busy waiting

        if preview:
            cv2.destroyWindow(window_name)
        print("Kinect feed stopped")
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def track(data_queue1: queue.Queue, data_queue2: queue.Queue, stream=True):
    global camera_poses
    print(camera_poses)
    print("Tracking started")
    
    if stream:
        HOST = "127.0.0.1"
        PORT = 5001
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
            if not (data_queue1.empty() or data_queue2.empty()):
                data1 = data_queue1.get_nowait()
                data2 = data_queue2.get_nowait()
                image_points = [data1, data2]
                object_points, image_p = find_point_correspondance_and_object_points(image_points, camera_poses, 4)
                
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


def run_single_camera(device_id, data_queue):
    """
    Initialize and run a single Kinect device.
    """
    try:
        pykinect.initialize_libraries(track_body=False)
        
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        kinect = pykinect.start_device(config=device_config)
        print(f'Kinect {device_id} initialized successfully')
        
        result = track_points(kinect, data_queue, preview=True)
        
        return result
        
    except Exception as ex:
        print(f'Error initializing Kinect {device_id}: {ex}')
        return False


def main():
    """
    Main entry point.
    """
    global running
    try:
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK')
        
        # Correct device detection
        device_count = pykinect.Device.device_get_installed_count()
        print(f'Number of Kinect devices detected: {device_count}')
        
        if device_count == 0:
            print('No Kinect devices detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        if device_count < 2:
            print('Warning: Only one Kinect device detected. Stereo tracking requires at least 2 devices.')
            print('Continuing with single device for testing...')
        
        data_queue1 = queue.Queue(maxsize=10)
        data_queue2 = queue.Queue(maxsize=10) if device_count >= 2 else queue.Queue(maxsize=10)

        process_thread = threading.Thread(target=track, args=(data_queue1, data_queue2))
        process_thread.daemon = True
        process_thread.start()
        
        camera1_thread = threading.Thread(target=run_single_camera, args=(0, data_queue1))
        camera1_thread.daemon = True
        camera1_thread.start()
        
        camera2_thread = None
        if device_count >= 2:
            camera2_thread = threading.Thread(target=run_single_camera, args=(1, data_queue2))
            camera2_thread.daemon = True
            camera2_thread.start()

        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            running.clear()
        
        print('Stopping cameras...')
        camera1_thread.join(timeout=2)
        if camera2_thread:
            camera2_thread.join(timeout=2)
        
        print('\nDone!')
        return True
        
    except Exception as ex:
        print(f'Error: {ex}')
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
