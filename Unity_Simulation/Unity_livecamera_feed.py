import os
import mmap
import numpy as np
from PIL import Image
import cv2
import time
import threading
import sys

# List your cameras (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

OUT_DIR = "Unity_Simulation/texture_import_from_unity"
os.makedirs(OUT_DIR, exist_ok=True)

# Global control variables (matching TakePhotos pattern)
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
        print(f"[{cam['name']}] MMF not found/openable: {e}")
        return None

def unity_to_opencv_image(cam, data):
    """Convert Unity texture data to OpenCV image format"""
    h, w, c = cam["height"], cam["width"], cam["channels"]
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
    
    # Unity → bottom-left origin; flip vertically for display
    arr_flipped = np.flip(arr, axis=0)
    
    # Convert RGBA to BGR for OpenCV (drop alpha channel)
    if c == 4:
        # Assuming RGBA format, convert to BGR
        bgr_image = cv2.cvtColor(arr_flipped, cv2.COLOR_RGBA2BGR)
    else:
        bgr_image = arr_flipped
    
    return bgr_image

def acquire_and_display_images(cam, cam_num, flipped=True, floor=False):
    """
    Continuously acquire images from Unity camera and display them using OpenCV.
    Similar to the PySpin camera function but for Unity cameras.
    """
    global running, take_photo
    
    window_name = f'Unity Camera Feed - {cam["name"]}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Frame rate calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print(f'Acquiring images from {cam["name"]}...')
    
    # Main acquisition loop
    while running.is_set():
        try:
            # Read data from Unity camera
            data = read_camera_data(cam)
            
            if data is None:
                time.sleep(0.1)  # Wait a bit if no data available
                continue
                
            # Check if we have actual image data (not all zeros)
            nonzero = int(np.count_nonzero(np.frombuffer(data, dtype=np.uint8)))
            
            if nonzero == 0:
                # Display black frame with message
                black_frame = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)
                cv2.putText(black_frame, "Waiting for Unity...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, black_frame)
            else:
                # Convert Unity data to OpenCV image
                image_data = unity_to_opencv_image(cam, data)
                
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the image
                cv2.imshow(window_name, image_data_flipped)
                
                # Handle photo taking
                if take_photo.is_set():
                    print(f'Taking photo...{cam_num}')
                    if floor:
                        os.makedirs(f'Unity_Simulation/tracking/cam{cam_num}', exist_ok=True)
                        image_name = f'Unity_Simulation/tracking/cam{cam_num}/{int(time.time())}.png'
                    else:
                        os.makedirs(f'Unity_Simulation/captured_images/cam{cam_num}', exist_ok=True)
                        image_name = f'Unity_Simulation/captured_images/cam{cam_num}/{int(time.time())}.png'
                    cv2.imwrite(image_name, image_data)
                    take_photo.clear()
            
            # Process OpenCV GUI events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print('ESC pressed. Exiting...')
                running.clear()
                break
            elif key == ord('s'):  # 's' key for saving image
                take_photo.set()
                
        except Exception as ex:
            print(f'Error reading from {cam["name"]}: {ex}')
            time.sleep(0.1)
    
    # Clean up
    cv2.destroyWindow(window_name)
    print(f"Unity camera feed stopped for {cam['name']}")

def run_single_camera(cam, cam_num=0, flipped=True, floor=True):
    """
    Unity camera execution function.
    Similar to the PySpin camera function but for Unity cameras.
    """
    try:
        # Run acquisition and display function
        acquire_and_display_images(cam, cam_num, flipped, floor)
        return True
        
    except Exception as ex:
        print(f'Error with Unity camera {cam["name"]}: {ex}')
        return False

def main(auto=False, floor=False, flipped=True):
    """
    Main function - similar structure to TakePhotos_dual-Extrinsics.py
    """
    global running, take_photo
    
    try:
        print('Unity MoCap v2.0')
        print(f'Number of Unity cameras configured: {len(cameras)}')
        
        # Check if we have cameras configured
        if len(cameras) == 0:
            print('No Unity cameras configured!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        # Start camera threads
        camera_threads = []
        for i, cam in enumerate(cameras):
            camera_thread = threading.Thread(
                target=run_single_camera, 
                args=(cam, i, flipped, floor)
            )
            camera_thread.start()
            camera_threads.append(camera_thread)
        
        time.sleep(2)  # Allow cameras to initialize
        
        # Main control loop
        while running.is_set():
            time.sleep(0.5)
            if auto:
                take_photo.set()
                time.sleep(0.5)
                take_photo.clear()
        
        time.sleep(1)
        print('Stopping Unity cameras...')
        
        # Wait for all camera threads to finish
        for thread in camera_threads:
            thread.join()
        
        print('\nDone!')
        return True
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False

if __name__ == '__main__':
    success = main(auto=False, floor=False, flipped=True)
    print('Exiting...')
    sys.exit(0 if success else 1)
