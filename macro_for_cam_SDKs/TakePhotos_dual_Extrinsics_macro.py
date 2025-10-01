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
from pypreprocessor import pypreprocessor

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

#exclude
# run the script in 'FLIR' mode
if 'FLIR' in sys.argv:
    pypreprocessor.defines.append('FLIR')

# run the script in 'Azure' mode
if 'Azure' in sys.argv:
    pypreprocessor.defines.append('Azure')

# run the script in 'Unity' mode
if 'Unity' in sys.argv:
    pypreprocessor.defines.append('Unity')

# run the script in 'EZVIZ' mode
if 'EZVIZ' in sys.argv:
    pypreprocessor.defines.append('EZVIZ')

pypreprocessor.parse()

#endexclude
#ifdef FLIR
def acquire_and_display_images_flir(cam, nodemap, nodemap_tldevice, cam_num, flipped=True, floor=False):
    """
    This function continuously acquires images from a device and displays them using OpenCV.
    
    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global running
    try:
        # Configure stream buffer handling mode
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode. Aborting...')
            return False
            
        # Set buffer handling mode to NewestOnly for lowest latency
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode. Aborting...')
            return False
            
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())
        
        # Configure acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False
            
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False
            
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        print('Acquisition mode set to continuous...')
        
        # Retrieve device serial number for window name
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print(f'Device serial number retrieved as {device_serial_number}...')
        
        window_name = f'Camera Feed - {device_serial_number}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Configure exposure
        try:
            # Set exposure auto to off for manual control
            node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            if PySpin.IsWritable(node_exposure_auto):
                node_exposure_auto.SetIntValue(node_exposure_auto.GetEntryByName('Off').GetValue())
                
                # Set exposure time manually (in microseconds)
                node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
                if PySpin.IsWritable(node_exposure_time):
                    exposure_time = 5000.0  # 5ms exposure, adjust as needed
                    node_exposure_time.SetValue(exposure_time)
                    print(f'Exposure time set to {exposure_time} us')
        except PySpin.SpinnakerException as ex:
            print(f'Error setting exposure: {ex}')
        
        # Begin acquiring images
        cam.BeginAcquisition()
        print('Acquiring images...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)  # Allow time for camera to stabilize
        
        # Main acquisition loop
        while running.is_set():
            try:
                # Get next image with shorter timeout for responsiveness
                image_result = cam.GetNextImage(500)
                
                if not image_result.IsIncomplete():
                    # Get image data as numpy array and optimize display
                    image_data = image_result.GetNDArray().copy()
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GR2BGR)  # Convert to BGR
                    if flipped:
                        image_data_flipped = cv2.flip(image_data, 1)
                    else:
                        image_data_flipped = image_data
                    
                    # Calculate and display FPS every second
                    frame_count += 1
                    if frame_count % 30 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        frame_count = 0
                        start_time = end_time
                    
                    # Add FPS text to the image
                    cv2.putText(image_data_flipped, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display the image using OpenCV (much faster than matplotlib)
                    cv2.imshow(window_name, image_data_flipped)
                    
                    # Process any OpenCV GUI events (like window closing)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break
                    if key == ord('s'):  # 's' key for saving image
                        take_photo.set()
                    if take_photo.is_set():
                        print(f'Taking photo...{cam_num}')
                        if floor:
                            image_name = f'macro_for_cam_SDKs/FLIR/tracking/cam{cam_num}/{int(time.time())}.png'
                        else:
                            image_name = f'macro_for_cam_SDKs/FLIR/captured_images/cam{cam_num}/{int(time.time())}.png'
                        cv2.imwrite(image_name, image_data)
                        take_photo.clear() # idk why but it works
                
                # Release the image to avoid buffer filling
                image_result.Release()
                
            except PySpin.SpinnakerException as ex:
                print(f'Error: {ex}')
                running.clear()
        
        # End acquisition and clean up
        cam.EndAcquisition()
        cv2.destroyWindow(window_name)
        print("Camera feed stopped")
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def run_single_camera_flir(cam,cam_num=0,flipped=True,floor=True):
    """
    Camera initialization and execution function.
    
    :param cam: Camera to run on.
    :param cam_num: Camera number for saving images.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Initialize camera
        cam.Init()
        
        # Retrieve nodemap
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        nodemap = cam.GetNodeMap()
            
        # Run acquisition and display function
        result = acquire_and_display_images_flir(cam, nodemap, nodemap_tldevice,cam_num,flipped,floor)
        
        # Deinitialize camera
        cam.DeInit()
        return result
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False


def main_flir(auto=False,floor=False,flipped=True):
    """
    Main function.
    """
    global running
    try:
        # Get system instance
        system = PySpin.System.GetInstance()
        
        # Print system info
        # version = system.GetLibraryVersion()
        print(f'MoCap v2.0')
        
        # Get camera list
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'Number of cameras detected: {num_cameras}')
        
        # Check if cameras are available
        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            print('No cameras detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
            
        camera1_display = threading.Thread(target=run_single_camera_flir, args=(cam_list[0],0,flipped,floor,))
        camera1_display.start()
        camera2_display = threading.Thread(target=run_single_camera_flir, args=(cam_list[1],1,flipped,floor,))
        camera2_display.start()

        time.sleep(8)

        while running.is_set():
            time.sleep(0.5)
            if auto:
                take_photo.set()
                time.sleep(0.5)
                take_photo.clear()
        
        time.sleep(1)
        print('Stopping cameras...')
        camera1_display.join()
        camera2_display.join()

        # Clean up
        # del cam_list[0]  # Important for proper memory management
        cam_list.Clear()
        system.ReleaseInstance()
        
        print('\nDone!')
        return True
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False


if __name__ == '__main__':
    success = main_flir(auto=False,floor=False,flipped=True)
    print('Exiting...')
    sys.exit(0 if success else 1)

#else
#ifdef Azure
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


if __name__ == '__main__':
    try:
        success = main_azure(auto=False, floor=False, flipped=True)
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

#else
#ifdef Unity
# Unity camera configuration (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

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

#else
#ifdef EZVIZ

# Change this to your Ezviz H3C RTSP URL
# Format: rtsp://admin:VERIFICATION_CODE@<IP>:<PORT>/h264
CAMERA_STREAMS = [
    "rtsp://admin:WOJWUD@169.254.27.194:554/h264",  # Camera 1
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


if __name__ == '__main__':
    try:
        success = main_ezviz(auto=False, floor=False, flipped=True)
        print('Exiting...')
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

#endifall