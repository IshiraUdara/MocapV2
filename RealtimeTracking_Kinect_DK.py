import sys
import time
import threading
import cv2
import numpy as np
from pykinect_azure import pykinect
from pykinect_azure.k4a import *
from lib.ImageOperations import _find_dot
from lib.Helpers import find_point_correspondance_and_object_points, get_extrinsics
import queue
import socket
import msgpack
import pyk4a
from pyk4a import PyK4A, Config

running = threading.Event()
running.set()
camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")

def track_points(kinect, data_queue: queue.Queue, preview=False):
    """
    This function continuously acquires images from Azure Kinect DK and processes them.
    
    :param kinect: Azure Kinect device instance.
    :param data_queue: Queue to store detected points.
    :param preview: Whether to show preview window.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global running
    try:
        # Get device serial number for window identification
        device_serial_number = kinect.get_serialnum()
        print(f'Device serial number: {device_serial_number}')
        
        window_name = f'Kinect Feed - {device_serial_number}'
        if preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print('Starting image acquisition...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)  # Allow time for camera to stabilize
        
        # Main acquisition loop
        while running.is_set():
            try:
                # Get capture from Kinect
                capture = kinect.get_capture()
                
                if capture.color is not None:
                    # Get color image as numpy array
                    color_image = capture.color
                    
                    # Convert to OpenCV format (BGR)
                    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                    
                    # Convert to grayscale for dot detection
                    gray_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2GRAY)
                    
                    # Find dots in the image
                    processed_image, detected_points = _find_dot(gray_image, print_location=True)

                    try:
                        # Drain the queue to get the most recent data
                        if data_queue.full():
                            data_queue.get_nowait()
                        data_queue.put_nowait(detected_points)
                    except queue.Full:
                        print("Queue is full")
                    
                    # Calculate and display FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        frame_count = 0
                        start_time = end_time
                    
                    if preview:
                        # Add FPS text to the image
                        cv2.putText(processed_image, f"FPS: {fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Display the image
                        cv2.imshow(window_name, processed_image)
                        
                        # Process any OpenCV GUI events
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            print('ESC pressed. Exiting...')
                            running.clear()
                            break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as ex:
                print(f'Error during capture: {ex}')
                continue
        
        # Clean up
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
        PORT = 5000
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
            pass  # Continue if queue is empty
        except Exception as ex:
            print(f"Tracking error: {ex}")
            
        time.sleep(0.01)


def run_single_camera(device_id, data_queue):
    """
    Kinect initialization and execution function.
    
    :param device_id: Kinect device ID (0 for first device, 1 for second, etc.)
    :param data_queue: Queue to store detected points
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        # Modify configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        
        # Start device
        kinect = pykinect.start_device(device_id=device_id, config=device_config)
        print(f'Kinect {device_id} initialized successfully')
        
        # Run tracking function
        result = track_points(kinect, data_queue, preview=True)
        
        return result
        
    except Exception as ex:
        print(f'Error initializing Kinect {device_id}: {ex}')
        return False


def main():
    """
    Main function.
    """
    global running
    try:
        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK')
        
        # Check for available devices
        device_count = 0
        print(f'Number of Kinect devices detected: {device_count}')
        
        # Check if devices are available
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

        # Start tracking thread
        process_thread = threading.Thread(target=track, args=(data_queue1, data_queue2))
        process_thread.daemon = True
        process_thread.start()
        
        # Start camera threads
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
        
        # Wait for threads to finish
        print('Stopping cameras...')
        camera1_thread.join(timeout=2)
        if camera2_thread:
            camera2_thread.join(timeout=2)
        
        print('\nDone!')
        return True
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False


# Try to detect devices using pyk4a
device_count = 0
try:
    for i in range(4):
        try:
            # Try to create a PyK4A instance
            k4a = PyK4A(device_id=i)
            k4a.start()
            device_count += 1
            print(f'Found device {i}')
            k4a.stop()
        except Exception as e:
            print(f'No device found at ID {i}')
            break
except Exception as e:
    print(f'Error during device detection: {e}')

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