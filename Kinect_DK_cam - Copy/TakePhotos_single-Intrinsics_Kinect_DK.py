import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import *
from pykinect_azure.k4a import _k4a

running = threading.Event()
running.set()

def acquire_and_display_images(kinect):
    """
    This function continuously acquires images from Azure Kinect DK and displays them using OpenCV.
    
    Parameters:
        kinect: Azure Kinect device instance.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    global running
    try:
        # Retrieve device serial number for window name
        device_serial_number = kinect.get_serialnum()
        print(f'Device serial number retrieved as {device_serial_number}...')
        
        window_name = f'Kinect Feed - {device_serial_number}'
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
                        
                        # Flip image horizontally for mirror effect
                        flipped_image_data = cv2.flip(color_image_bgr, 1)
                        
                        # Calculate and display FPS every second
                        frame_count += 1
                        if frame_count % 30 == 0:
                            end_time = time.time()
                            fps = frame_count / (end_time - start_time)
                            frame_count = 0
                            start_time = end_time
                        
                        # Add FPS text to the image
                        cv2.putText(flipped_image_data, f"FPS: {fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Display the image using OpenCV
                        cv2.imshow(window_name, flipped_image_data)
                        
                        # Process any OpenCV GUI events
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC key
                            print('ESC pressed. Exiting...')
                            running.clear()
                            break
                        if key == ord('s'):  # 's' key for saving image
                            print('Taking photo...')
                            image_name = f'Kinect_DK_cam/captured_images/captured_image_{int(time.time())}_{device_serial_number}.png'
                            # Create directory if it doesn't exist
                            os.makedirs(f'Kinect_DK_cam/captured_images', exist_ok=True)
                            cv2.imwrite(image_name, color_image_bgr)  # Save original image without flip
                            print(f'Image saved as: {image_name}')
                        if key == ord('q'):  # 'q' key for quit
                            print('Q pressed. Exiting...')
                            running.clear()
                            break
                            
            except Exception as ex:
                print(f'Error during capture: {ex}')
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        cv2.destroyWindow(window_name)
        print("Kinect feed stopped")
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def run_single_camera(device_id=0):
    """
    Initialize and run a single Kinect device.
    
    Parameters:
        device_id (int): Device ID to initialize.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        pykinect.initialize_libraries(track_body=False)
        
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        kinect = pykinect.start_device(config=device_config)
        print(f'Kinect {device_id} initialized successfully')
        
        # Run acquisition and display function
        result = acquire_and_display_images(kinect)
        
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
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK Photo Capture')
        
        # Correct device detection
        device_count = pykinect.Device.device_get_installed_count()
        print(f'Number of Kinect devices detected: {device_count}')
        
        # Check if cameras are available
        if device_count == 0:
            print('No Kinect devices detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        print('Press "s" to save image, "q" or ESC to quit')
        
        camera1_display = threading.Thread(target=run_single_camera, args=(0,))
        camera1_display.daemon = True
        camera1_display.start()

        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            running.clear()
        
        print('Stopping cameras...')
        camera1_display.join(timeout=2)
        
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