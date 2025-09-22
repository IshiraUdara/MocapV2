import sys
import time
import threading
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import *
from pykinect_azure.k4a import _k4a
import PySpin
import os
import mmap
import numpy as np

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

def acquire_and_display_images_azure(kinect, cam_num, flipped=True, floor=False):
    """
    This function continuously acquires images from Azure Kinect DK and displays them using OpenCV.
    
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
        
        window_name = f'Kinect Feed - {device_serial_number}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print('Starting image acquisition...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)  # Allow time for camera to stabilize
        
        # Create directories if they don't exist
        if floor:
            os.makedirs(f'macro_for_cam_SDKs/Kinect_DK_cam/tracking/cam{cam_num}', exist_ok=True)
        else:
            os.makedirs(f'macro_for_cam_SDKs/Kinect_DK_cam/captured_images/cam{cam_num}', exist_ok=True)

        # Main acquisition loop
        while running.is_set():
            try:
                # Get capture from Kinect
                capture = kinect.get_capture()
                
                if capture is not None:
                    # Get color image using the k4a function
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
                                
                                # Convert from BGRA to BGR
                                if color_image.shape[2] == 4:  # BGRA
                                    image_data = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                                else:  # Already BGR
                                    image_data = color_image
                                
                                # Apply horizontal flip if requested
                                if flipped:
                                    image_data_flipped = cv2.flip(image_data, 1)
                                else:
                                    image_data_flipped = image_data
                                
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
                        print(f'Taking photo...{cam_num}')
                        if floor:
                            image_name = f'macro_for_cam_SDKs/Kinect_DK_cam/tracking/cam{cam_num}/{int(time.time())}.png'
                        else:
                            image_name = f'macro_for_cam_SDKs/Kinect_DK_cam/captured_images/cam{cam_num}/{int(time.time())}.png'
                        cv2.imwrite(image_name, image_data)
                        take_photo.clear()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as ex:
                print(f'Error during capture: {ex}')
                continue
        
        # Clean up
        cv2.destroyWindow(window_name)
        print("Kinect feed stopped")
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def run_single_camera_azure(device_id, cam_num=0, flipped=True, floor=True):
    """
    Kinect initialization and execution function.
    
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
        
        # Configure device
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        
        # Start device
        kinect = pykinect.start_device(config=device_config)
        print(f'Kinect {device_id} initialized successfully')
        
        # Run acquisition and display function
        result = acquire_and_display_images_azure(kinect, cam_num, flipped, floor)
        
        return result
        
    except Exception as ex:
        print(f'Error initializing Kinect {device_id}: {ex}')
        return False


def main_azure(auto=False, floor=False, flipped=True):
    """
    Main function.
    """
    print('Azure Kinect DK mode activated')
    global running
    try:
        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK')
        
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
        camera1_display = threading.Thread(target=run_single_camera_azure, args=(0, 0, flipped, floor))
        camera1_display.daemon = True
        camera1_display.start()
        
        camera2_display = None
        if device_count >= 2:
            camera2_display = threading.Thread(target=run_single_camera_azure, args=(1, 1, flipped, floor))
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
        print('Stopping cameras...')
        
        # Wait for threads to finish
        camera1_display.join(timeout=2)
        if camera2_display:
            camera2_display.join(timeout=2)
        
        print('\nDone!')
        return True
        
    except Exception as ex:
        print(f'Error: {ex}')
        return False

# Change this to your Ezviz H3C RTSP URL
# Format: rtsp://admin:VERIFICATION_CODE@<IP>:<PORT>/h264
CAMERA_STREAMS = [
    "rtsp://admin:WOJWUD@192.168.1.112:554/h264",  # Camera 1
    # Add more streams here if you have multiple Ezviz cameras
]

def acquire_and_display_images_ezviz(rtsp_url, cam_num, flipped=True, floor=False):
    """
    Continuously acquires images from an Ezviz H3C camera using RTSP and displays them.

    :param rtsp_url: RTSP URL of the Ezviz camera
    :param cam_num: Camera number for saving images
    :param flipped: Whether to flip the image horizontally
    :param floor: Whether to save in tracking folder or captured_images folder
    """
    global running, take_photo
    try:
        window_name = f'Ezviz H3C Feed - Cam{cam_num}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print(f'Starting Ezviz H3C feed on {rtsp_url}')

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"Failed to open stream for camera {cam_num}")
            return False

        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)

        # Create directories
        if floor:
            os.makedirs(f'macro_for_cam_SDKs/Ezviz_H3C_cam/tracking/cam{cam_num}', exist_ok=True)
        else:
            os.makedirs(f'macro_for_cam_SDKs/Ezviz_H3C_cam/captured_images/cam{cam_num}', exist_ok=True)

        while running.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {cam_num}: No frame received")
                time.sleep(0.1)
                continue

            # Flip if requested
            if flipped:
                frame = cv2.flip(frame, 1)

            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = end_time

            # Add FPS text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the feed
            cv2.imshow(window_name, frame)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print('ESC pressed. Exiting...')
                running.clear()
                break
            elif key == ord('s'):  # save photo
                take_photo.set()

            # Save photo if triggered
            if take_photo.is_set():
                print(f'Taking photo... cam{cam_num}')
                if floor:
                    image_name = f'macro_for_cam_SDKs/Ezviz_H3C_cam/tracking/cam{cam_num}/{int(time.time())}.png'
                else:
                    image_name = f'macro_for_cam_SDKs/Ezviz_H3C_cam/captured_images/cam{cam_num}/{int(time.time())}.png'
                cv2.imwrite(image_name, frame)
                take_photo.clear()

            time.sleep(0.001)

        cap.release()
        cv2.destroyWindow(window_name)
        print(f"Ezviz H3C cam{cam_num} feed stopped")

    except Exception as ex:
        print(f'Error in cam{cam_num}: {ex}')
        return False

    return True


def run_single_camera_ezviz(rtsp_url, cam_num=0, flipped=True, floor=True):
    """Initialize and run a single Ezviz camera feed"""
    try:
        result = acquire_and_display_images_ezviz(rtsp_url, cam_num, flipped, floor)
        return result
    except Exception as ex:
        print(f'Error running Ezviz cam{cam_num}: {ex}')
        return False


def main_ezviz(auto=False, floor=False, flipped=True):
    """Main function for Ezviz cameras"""
    print('Ezviz H3C mode activated')
    global running
    try:
        print(f'MoCap v2.0 - Ezviz H3C')
        device_count = len(CAMERA_STREAMS)
        print(f'Number of Ezviz cameras configured: {device_count}')

        if device_count == 0:
            print('No Ezviz cameras configured!')
            sys.exit(0)

        # Start camera threads
        threads = []
        for idx, rtsp_url in enumerate(CAMERA_STREAMS):
            t = threading.Thread(target=run_single_camera_ezviz, args=(rtsp_url, idx, flipped, floor))
            t.daemon = True
            t.start()
            threads.append(t)

        time.sleep(5)  # Let cameras initialize

        # Main loop
        while running.is_set():
            time.sleep(0.5)
            if auto:
                take_photo.set()
                time.sleep(0.5)
                take_photo.clear()

        print('Stopping cameras...')
        for t in threads:
            t.join(timeout=2)

        print('\nDone!')
        return True

    except Exception as ex:
        print(f'Error: {ex}')
        return False

def main_combined(auto=False, floor=False, flipped=True):
    """Main function for Ezviz cameras"""
    print('Ezviz H3C mode activated')
    global running
    try:
        print(f'MoCap v2.0 - Ezviz H3C')
        device_count = len(CAMERA_STREAMS)
        print(f'Number of Ezviz cameras configured: {device_count}')

        if device_count == 0:
            print('No Ezviz cameras configured!')
            sys.exit(0)

        # Start camera threads
        threads = []
        for idx, rtsp_url in enumerate(CAMERA_STREAMS):
            t = threading.Thread(target=run_single_camera_ezviz, args=(rtsp_url, idx, flipped, floor))
            t.daemon = True
            t.start()
            threads.append(t)
  
        print('Azure Kinect DK mode activated')

        # Initialize pykinect
        pykinect.initialize_libraries(track_body=False)
        
        print(f'MoCap v2.0 - Azure Kinect DK')
        
        # Try to detect available devices
        device_count_azure = 1  # Assume 1 device for now since detection is unreliable
        print(f'Number of Kinect devices detected: {device_count_azure}')

        # Check if devices are available
        if device_count_azure == 0:
            print('No Kinect devices detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        # if device_count < 2:
        #     print('Warning: Only one Kinect device detected. Dual camera setup requires at least 2 devices.')
        #     print('Continuing with single device for testing...')
        
        # Start camera threads
        camera1_display = threading.Thread(target=run_single_camera_azure, args=(0, 0, flipped, floor))
        camera1_display.daemon = True
        camera1_display.start()
        
        camera2_display = None
        if device_count >= 2:
            camera2_display = threading.Thread(target=run_single_camera_azure, args=(1, 1, flipped, floor))
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

        print('Stopping cameras...')
        for t in threads:
            t.join(timeout=2)
        
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
        success = main_combined(auto=False, floor=False, flipped=True)
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
