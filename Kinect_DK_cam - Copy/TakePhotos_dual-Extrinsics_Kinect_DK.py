import sys
import time
import threading
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import *
from pykinect_azure.k4a import _k4a
import os
import numpy as np

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

def acquire_and_display_images(kinect, cam_num, flipped=True, floor=False):
    """
    This function continuously acquires IR images from Azure Kinect DK and displays them using OpenCV.
    
    :param kinect: Azure Kinect device instance.
    :param cam_num: Camera number for saving images.
    :param flipped: Whether to flip the image horizontally.
    :param floor: Whether to save in tracking folder or captured_images folder.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global running, take_photo
    try:
        # Get device serial number for window identification
        device_serial_number = kinect.get_serialnum()
        print(f'Device serial number: {device_serial_number}')
        
        window_name = f'Kinect IR Feed - {device_serial_number}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print('Starting IR image acquisition...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)  # Allow time for camera to stabilize
        
        # Create directories if they don't exist
        if floor:
            os.makedirs(f'Kinect_DK_cam/tracking/cam{cam_num}', exist_ok=True)
        else:
            os.makedirs(f'Kinect_DK_cam/captured_images/cam{cam_num}', exist_ok=True)

        # Main acquisition loop
        while running.is_set():
            try:
                # Get capture from Kinect
                capture = kinect.get_capture()
                
                if capture is not None:
                    # Get IR image using the k4a function
                    ir_image_handle = _k4a.k4a_capture_get_ir_image(capture)
                    if ir_image_handle:
                        try:
                            # Get image properties first
                            width = _k4a.k4a_image_get_width_pixels(ir_image_handle)
                            height = _k4a.k4a_image_get_height_pixels(ir_image_handle)
                            buffer_size = _k4a.k4a_image_get_size(ir_image_handle)
                            
                            # Check if we have valid dimensions
                            if width > 0 and height > 0 and buffer_size > 0:
                                # Convert to numpy array using the Image class
                                ir_image_result = Image(ir_image_handle).to_numpy()

                                # Check if the result is a tuple (success, array)
                                if isinstance(ir_image_result, tuple):
                                    success, ir_image = ir_image_result
                                    if not success or ir_image is None:
                                        print("Failed to convert IR image to numpy array")
                                        _k4a.k4a_image_release(ir_image_handle)
                                        continue
                                else:
                                    ir_image = ir_image_result
                                
                                # IR image is typically 16-bit, convert to 8-bit for display
                                if ir_image.dtype == np.uint16:
                                    # Normalize to 8-bit range for display
                                    ir_image_8bit = (ir_image / 256).astype(np.uint8)
                                else:
                                    ir_image_8bit = ir_image
                                
                                # Convert grayscale to BGR for OpenCV display
                                if len(ir_image_8bit.shape) == 2:  # Grayscale
                                    image_data = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)
                                else:
                                    image_data = ir_image_8bit
                                
                                # Apply horizontal flip if requested
                                if flipped:
                                    image_data_flipped = cv2.flip(image_data, 1)
                                else:
                                    image_data_flipped = image_data
                                
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
                        continue
                    
                    # Calculate and display FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        frame_count = 0
                        start_time = end_time
                    
                    # Add FPS text to the image
                    cv2.putText(image_data_flipped, f"IR FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display the image using OpenCV
                    cv2.imshow(window_name, image_data_flipped)
                    
                    # Process any OpenCV GUI events
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break
                    elif key == ord('s'):  # 's' key for saving image
                        take_photo.set()
                    
                    # Check if photo should be taken
                    if take_photo.is_set():
                        print(f'Taking IR photo...{cam_num}')
                        if floor:
                            image_name = f'Kinect_DK_cam/tracking/cam{cam_num}/{int(time.time())}_ir.png'
                        else:
                            image_name = f'Kinect_DK_cam/captured_images/cam{cam_num}/{int(time.time())}_ir.png'
                        # Save the original IR data (before conversion to 8-bit for display)
                        if ir_image.dtype == np.uint16:
                            cv2.imwrite(image_name, ir_image_8bit)  # Save 8-bit version
                        else:
                            cv2.imwrite(image_name, ir_image)
                        take_photo.clear()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as ex:
                print(f'Error during IR capture: {ex}')
                continue
        
        # Clean up
        cv2.destroyWindow(window_name)
        print("Kinect IR feed stopped")
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def run_single_camera(device_id, cam_num=0, flipped=True, floor=True):
    """
    Kinect initialization and execution function for IR capture.
    
    :param device_id: Kinect device ID (0 for first device, 1 for second, etc.)
    :param cam_num: Camera number for saving images.
    :param flipped: Whether to flip the image horizontally.
    :param floor: Whether to save in tracking folder or captured_images folder.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        # Configure device for IR capture
        device_config = pykinect.default_configuration
        # Disable color camera to save bandwidth and focus on IR
        #device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_OFF
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        # Enable depth mode to get IR data
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        
        # Start device
        kinect = pykinect.start_device(config=device_config)
        print(f'Kinect {device_id} initialized successfully for IR capture')
        
        # Run acquisition and display function
        result = acquire_and_display_images(kinect, cam_num, flipped, floor)
        
        return result
        
    except Exception as ex:
        print(f'Error initializing Kinect {device_id} for IR: {ex}')
        return False


def main(auto=False, floor=False, flipped=True):
    """
    Main function for IR capture.
    """
    global running
    try:
        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK IR Capture')
        
        # Try to detect available devices
        device_count = 1  # Assume 1 device for now since detection is unreliable
        print(f'Number of Kinect devices detected: {device_count}')
        
        # Check if devices are available
        if device_count == 0:
            print('No Kinect devices detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        if device_count < 2:
            print('Warning: Only one Kinect device detected. Dual camera setup requires at least 2 devices.')
            print('Continuing with single device for testing...')
        
        # Start camera threads
        camera1_display = threading.Thread(target=run_single_camera, args=(0, 0, flipped, floor))
        camera1_display.daemon = True
        camera1_display.start()
        
        camera2_display = None
        if device_count >= 2:
            camera2_display = threading.Thread(target=run_single_camera, args=(1, 1, flipped, floor))
            camera2_display.daemon = True
            camera2_display.start()

        time.sleep(8)  # Allow cameras to initialize

        # Main loop
        while running.is_set():
            time.sleep(0.5)
            if auto:
                take_photo.set()
                time.sleep(0.5)
                take_photo.clear()
        
        time.sleep(1)
        print('Stopping IR cameras...')
        
        # Wait for threads to finish
        camera1_display.join(timeout=2)
        if camera2_display:
            camera2_display.join(timeout=2)
        
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