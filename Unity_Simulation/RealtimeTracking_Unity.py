import sys
import os
import time
import threading
import cv2
import numpy as np
import mmap

# Add parent directory to Python path to find the lib module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import with CUDA, fallback to CPU if needed
_find_dot = None
use_cuda = False

try:
    from lib.ImageOperations import _find_dot
    use_cuda = True
    print("Using CUDA-accelerated dot detection")
except Exception as cuda_error:
    print(f"CUDA error: {cuda_error}")
    print("Falling back to CPU-only mode...")
    use_cuda = False

from lib.Helpers import find_point_correspondance_and_object_points, get_extrinsics
import queue
import socket
import msgpack

# Unity camera configuration (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

running = threading.Event()
running.set()
camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")

# Global variables for display
display_queue1 = queue.Queue(maxsize=2)
display_queue2 = queue.Queue(maxsize=2)

def simple_find_dot_cpu(image, print_location=False):
    """Simple CPU-based dot detection as fallback"""
    # Convert to binary image using adaptive threshold for better detection
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_points = []
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Filter by area to avoid noise
        area = cv2.contourArea(contour)
        if area > 10 and area < 1000:  # Adjust these values based on your dot size
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_points.append([cx, cy])
                
                # Draw circle on result image
                cv2.circle(result_image, (cx, cy), 5, (0, 255, 0), 2)
                
                if print_location:
                    print(f"Detected point at: ({cx}, {cy})")
    
    return result_image, detected_points

def read_camera_data(cam):
    """Read data from Unity camera via memory-mapped file"""
    size = cam["width"] * cam["height"] * cam["channels"]
    try:
        # Windows named shared memory: tagname must match Unity's mmfName
        with mmap.mmap(-1, size, tagname=cam["name"], access=mmap.ACCESS_READ) as mmf:
            mmf.seek(0)
            data = mmf.read(size)
            return data
    except (FileNotFoundError, OSError) as e:
        # Don't print this error repeatedly
        return None

def unity_to_opencv_image(cam, data):
    """Convert Unity texture data to OpenCV grayscale image format"""
    h, w, c = cam["height"], cam["width"], cam["channels"]
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
    
    # Unity → bottom-left origin; flip vertically for display
    arr_flipped = np.flip(arr, axis=0)
    
    # Convert RGBA to BGR then to grayscale for tracking
    if c == 4:
        bgr_image = cv2.cvtColor(arr_flipped, cv2.COLOR_RGBA2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = cv2.cvtColor(arr_flipped, cv2.COLOR_BGR2GRAY)
    
    return gray_image

def display_handler(preview=True):
    """Separate thread to handle OpenCV display to prevent blocking"""
    global running, display_queue1, display_queue2
    
    if not preview:
        return
    
    # Create windows
    window1 = 'Unity Camera Feed - UnityCam1'
    window2 = 'Unity Camera Feed - UnityCam2'
    cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
    
    # Resize windows for better display
    cv2.resizeWindow(window1, 400, 400)
    cv2.resizeWindow(window2, 400, 400)
    
    print("Display handler started. Press ESC to exit.")
    
    while running.is_set():
        try:
            # Handle camera 1 display
            if not display_queue1.empty():
                image1 = display_queue1.get_nowait()
                cv2.imshow(window1, image1)
            
            # Handle camera 2 display
            if not display_queue2.empty():
                image2 = display_queue2.get_nowait()
                cv2.imshow(window2, image2)
            
            # Process OpenCV events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print('ESC pressed. Exiting...')
                running.clear()
                break
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        time.sleep(0.03)  # ~30 FPS display rate
    
    # Clean up
    cv2.destroyAllWindows()
    print("Display handler stopped")

def track_points_unity(cam, cam_num, data_queue: queue.Queue, display_queue: queue.Queue):
    """
    This function continuously acquires images from Unity camera and tracks points.
    
    :param cam: Unity camera configuration dict.
    :param cam_num: Camera number for identification.
    :param data_queue: Queue to put detected points.
    :param display_queue: Queue to put display images.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global running, _find_dot, use_cuda
    
    print(f'Starting point tracking for {cam["name"]}...')
    
    # Frame rate calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    last_mmf_error_time = 0
    
    # Main acquisition loop
    while running.is_set():
        try:
            # Read data from Unity camera
            data = read_camera_data(cam)
            
            if data is None:
                # Only print MMF error every 5 seconds to avoid spam
                current_time = time.time()
                if current_time - last_mmf_error_time > 5:
                    print(f"[{cam['name']}] Waiting for Unity memory-mapped file...")
                    last_mmf_error_time = current_time
                time.sleep(0.1)
                continue
                
            # Check if we have actual image data (not all zeros)
            nonzero = int(np.count_nonzero(np.frombuffer(data, dtype=np.uint8)))
            
            if nonzero == 0:
                # Create waiting message image
                black_frame = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)
                cv2.putText(black_frame, "Waiting for Unity...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Put in display queue if not full
                if not display_queue.full():
                    try:
                        display_queue.put_nowait(black_frame)
                    except queue.Full:
                        pass
                
                time.sleep(0.1)
                continue
            
            # Convert Unity data to OpenCV grayscale image
            image_data = unity_to_opencv_image(cam, data)
            
            # Find dots in the image using appropriate method
            if use_cuda and _find_dot is not None:
                image_data_with_dots, detected_points = _find_dot(image_data, print_location=False)
            else:
                image_data_with_dots, detected_points = simple_find_dot_cpu(image_data, print_location=False)
            
            # Put detected points in data queue
            try:
                # Drain the queue to get the most recent data
                while not data_queue.empty():
                    data_queue.get_nowait()
                data_queue.put_nowait(detected_points)
            except queue.Full:
                pass
            
            # Calculate and display FPS every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = end_time
            
            # Prepare display image
            display_image = image_data_with_dots.copy()
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            mode_text = "CUDA" if use_cuda else "CPU"
            cv2.putText(display_image, f"Mode: {mode_text}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_image, f"Points: {len(detected_points)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Put in display queue if not full
            if not display_queue.full():
                try:
                    display_queue.put_nowait(display_image)
                except queue.Full:
                    pass
            
        except Exception as ex:
            print(f'Error in {cam["name"]}: {ex}')
            time.sleep(0.1)
    
    print(f"Unity camera tracking stopped for {cam['name']}")
    return True

def track(data_queue1: queue.Queue, data_queue2: queue.Queue, stream=True):
    """Main tracking function that processes point data from both cameras"""
    global camera_poses
    print(camera_poses)
    print("Tracking started")
    
    connection = None
    server = None
    
    if stream:
        try:
            HOST = "127.0.0.1"
            PORT = 5001  # Different port to avoid conflict with IMU
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, PORT))
            server.listen(1)
            server.settimeout(1.0)  # Non-blocking accept
            print(f"Waiting for Unity to connect on port {PORT}...")
        except Exception as e:
            print(f"Error setting up server: {e}")
            stream = False
    
    point = [0, 0, 0, 0, 0, 0, 0, 0]
    fps = 0
    old_time = time.time()

    while running.is_set():
        try:
            # Try to accept connection if not connected
            if stream and connection is None and server is not None:
                try:
                    connection, _ = server.accept()
                    print("Unity connected!")
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"Connection error: {e}")
            
            fps = time.time() - old_time
            old_time = time.time()
            fps = 1 / fps if fps > 0 else 0
            
            if not (data_queue1.empty() or data_queue2.empty()):
                data1 = data_queue1.get_nowait()
                data2 = data_queue2.get_nowait()
                image_points = [data1, data2]
                object_points, image_p = find_point_correspondance_and_object_points(image_points, camera_poses, 4)
                
                if stream and connection:
                    if len(object_points) > 0:
                        point = object_points[0]
                        point = list(point)
                        point = [0, 0, 0, 0] + point
                    data_to_send = {"tracker1": point}
                    try:
                        connection.send(msgpack.packb(data_to_send, use_bin_type=True))
                        print(f"Object Points: {point}")
                    except (ConnectionResetError, BrokenPipeError):
                        print("\nUnity disconnected")
                        connection = None
                        continue
                else:
                    if len(object_points) > 0:
                        print(f"Object Points: {object_points}")
                
                if len(image_p) > 0:
                    print(f"Image Points: {image_p}")
                print(f"FPS: {fps:.2f}")
                
        except queue.Empty:
            pass  # No data available, continue
        except Exception as e:
            print(f"Error in tracking: {e}")
            
        time.sleep(0.01)
    
    # Clean up
    if connection:
        connection.close()
    if server:
        server.close()

def run_single_unity_camera(cam, cam_num, data_queue, display_queue):
    """
    Unity camera initialization and execution function.
    
    :param cam: Unity camera configuration dict.
    :param cam_num: Camera number.
    :param data_queue: Queue to put detected points.
    :param display_queue: Queue to put display images.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Run tracking function
        result = track_points_unity(cam, cam_num, data_queue=data_queue, display_queue=display_queue)
        return result
        
    except Exception as ex:
        print(f'Error with Unity camera {cam["name"]}: {ex}')
        return False

def main(preview=True):
    """
    Main function.
    """
    global running, display_queue1, display_queue2
    try:
        print('Unity MoCap v2.0')
        print(f'Number of Unity cameras configured: {len(cameras)}')
        
        # Check if we have cameras configured
        if len(cameras) < 2:
            print('Need at least 2 Unity cameras configured!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        # Create data queues
        data_queue1 = queue.Queue(maxsize=10)
        data_queue2 = queue.Queue(maxsize=10)

        # Start display handler thread if preview is enabled
        display_thread = None
        if preview:
            display_thread = threading.Thread(target=display_handler, args=(preview,))
            display_thread.daemon = True
            display_thread.start()

        # Start tracking thread
        process_thread = threading.Thread(target=track, args=(data_queue1, data_queue2, True))
        process_thread.daemon = True
        process_thread.start()
        
        # Start camera threads
        camera1_thread = threading.Thread(target=run_single_unity_camera, 
                                         args=(cameras[0], 0, data_queue1, display_queue1))
        camera1_thread.daemon = True
        camera1_thread.start()
        
        camera2_thread = threading.Thread(target=run_single_unity_camera, 
                                         args=(cameras[1], 1, data_queue2, display_queue2))
        camera2_thread.daemon = True
        camera2_thread.start()

        print("Unity MoCap system running. Press Ctrl+C to stop or ESC in preview window.")
        
        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('\nKeyboard interrupt. Stopping...')
            running.clear()
        
        time.sleep(1)
        print('Stopping Unity cameras...')
        
        print('\nDone!')
        return True
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False

if __name__ == '__main__':
    # Set preview=True to see camera feeds with detected points
    success = main(preview=True)
    print('Exiting...')
    sys.exit(0 if success else 1)