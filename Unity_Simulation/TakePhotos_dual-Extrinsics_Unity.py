import sys
import os
import time
import threading
import cv2
import numpy as np
import mmap

# Add parent directory to Python path to find the lib module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Unity camera configuration (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

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
    """Convert Unity texture data to OpenCV BGR image format"""
    h, w, c = cam["height"], cam["width"], cam["channels"]
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
    
    # Unity uses bottom-left origin; flip vertically for display
    arr_flipped = np.flip(arr, axis=0)
    
    # Convert RGBA to BGR for OpenCV
    if c == 4:
        bgr_image = cv2.cvtColor(arr_flipped, cv2.COLOR_RGBA2BGR)
    else:
        bgr_image = cv2.cvtColor(arr_flipped, cv2.COLOR_RGB2BGR)
    
    return bgr_image

def acquire_and_display_images(cam, cam_num, flipped=True, floor=False):
    """
    This function continuously acquires images from Unity camera and displays them using OpenCV.
    
    Parameters:
        cam: Unity camera configuration dict.
        cam_num: Camera number for identification.
        flipped: Whether to flip the image horizontally.
        floor: Whether to save to tracking folder (True) or captured_images folder (False).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    global running, take_photo
    try:
        # Create window
        window_name = f'Unity Camera Feed - {cam["name"]}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)  # Resize for better visibility
        
        print(f'Starting image acquisition for {cam["name"]}...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        last_mmf_error_time = 0
        time.sleep(1)  # Allow time for Unity to initialize
        
        # Create output directories
        if floor:
            output_dir = f'Unity_Simulation/tracking/cam{cam_num}'
        else:
            output_dir = f'Unity_Simulation/captured_images/cam{cam_num}'
        os.makedirs(output_dir, exist_ok=True)
        
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
                    
                    # Create waiting message image
                    waiting_frame = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)
                    cv2.putText(waiting_frame, "Waiting for Unity...", (10, cam["height"]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow(window_name, waiting_frame)
                    
                    key = cv2.waitKey(100)
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break
                    
                    continue
                
                # Check if we have actual image data (not all zeros)
                nonzero = int(np.count_nonzero(np.frombuffer(data, dtype=np.uint8)))
                
                if nonzero == 0:
                    # Create waiting message image
                    black_frame = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)
                    cv2.putText(black_frame, "Waiting for Unity data...", (10, cam["height"]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow(window_name, black_frame)
                    
                    key = cv2.waitKey(100)
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break
                    
                    continue
                
                # Convert Unity data to OpenCV BGR image
                image_data = unity_to_opencv_image(cam, data)
                
                # Apply horizontal flip if requested
                if flipped:
                    image_data_flipped = cv2.flip(image_data, 1)
                else:
                    image_data_flipped = image_data
                
                # Calculate and display FPS every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = end_time
                
                # Add FPS text to the image
                cv2.putText(image_data_flipped, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add camera info
                cv2.putText(image_data_flipped, f"Cam: {cam_num}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the image using OpenCV
                cv2.imshow(window_name, image_data_flipped)
                
                # Process any OpenCV GUI events
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print('ESC pressed. Exiting...')
                    running.clear()
                    break
                if key == ord('s'):  # 's' key for saving image
                    take_photo.set()
                
                # Handle photo taking
                if take_photo.is_set():
                    print(f'Taking photo... cam{cam_num}')
                    image_name = f'{output_dir}/{int(time.time())}.png'
                    # Save original image without flip and UI text
                    cv2.imwrite(image_name, image_data)
                    print(f'Image saved as: {image_name}')
                    take_photo.clear()
                
            except Exception as ex:
                print(f'Error during capture for {cam["name"]}: {ex}')
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        cv2.destroyWindow(window_name)
        print(f"Unity camera feed stopped for {cam['name']}")
        
    except Exception as ex:
        print(f'Error in {cam["name"]}: {ex}')
        return False
        
    return True

def run_single_camera(cam, cam_num=0, flipped=True, floor=True):
    """
    Unity camera initialization and execution function.
    
    Parameters:
        cam: Unity camera configuration dict.
        cam_num: Camera number for saving images.
        flipped: Whether to flip the image horizontally.
        floor: Whether to save to tracking folder (True) or captured_images folder (False).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Run acquisition and display function
        result = acquire_and_display_images(cam, cam_num, flipped, floor)
        return result
        
    except Exception as ex:
        print(f'Error with Unity camera {cam["name"]}: {ex}')
        return False

def main(auto=False, floor=False, flipped=True):
    """
    Main function.
    """
    global running, take_photo
    try:
        print('MoCap v2.0 - Unity Camera Photo Capture')
        print(f'Number of Unity cameras configured: {len(cameras)}')
        
        # Check if cameras are available
        if len(cameras) < 2:
            print('Need at least 2 Unity cameras configured!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        print('Controls:')
        print('  Press "s" to save image from both cameras')
        print('  Press ESC to quit')
        if auto:
            print('  Auto mode enabled - taking photos every 0.5 seconds')
        
        # Start camera threads
        camera1_display = threading.Thread(target=run_single_camera, 
                                          args=(cameras[0], 0, flipped, floor))
        camera1_display.daemon = True
        camera1_display.start()
        
        camera2_display = threading.Thread(target=run_single_camera, 
                                          args=(cameras[1], 1, flipped, floor))
        camera2_display.daemon = True
        camera2_display.start()

        time.sleep(2)  # Allow cameras to initialize

        # Auto photo mode or manual control
        if auto:
            while running.is_set():
                time.sleep(0.5)
                take_photo.set()
                time.sleep(0.1)  # Brief delay to ensure photo is taken
                take_photo.clear()
                time.sleep(0.4)  # Wait before next photo
        else:
            try:
                while running.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping...")
                running.clear()
        
        time.sleep(1)
        print('Stopping Unity cameras...')
        camera1_display.join(timeout=3)
        camera2_display.join(timeout=3)
        
        print('\nDone!')
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