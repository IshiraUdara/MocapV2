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
import PySpin
import numpy as np
import mmap

from pypreprocessor import pypreprocessor

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

pypreprocessor.parse()

#endexclude
#ifdef FLIR

def track_points(cam, nodemap, nodemap_tldevice,data_queue:queue.Queue,preview=False):
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
        # try:
        #     # Set exposure auto to off for manual control
        #     node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        #     if PySpin.IsWritable(node_exposure_auto):
        #         node_exposure_auto.SetIntValue(node_exposure_auto.GetEntryByName('Off').GetValue())
                
        #         # Set exposure time manually (in microseconds)
        #         node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        #         if PySpin.IsWritable(node_exposure_time):
        #             exposure_time = 5000.0  # 5ms exposure, adjust as needed
        #             node_exposure_time.SetValue(exposure_time)
        #             print(f'Exposure time set to {exposure_time} us')
        # except PySpin.SpinnakerException as ex:
        #     print(f'Error setting exposure: {ex}')
        
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
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GR2BGR)
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                    image_data, detected_points = _find_dot(image_data,print_location=True)

                    try:
                        # Drain the queue to get the most recent data
                        if data_queue.full():
                            data_queue.get_nowait()
                        data_queue.put_nowait(detected_points)
                    except queue.Empty:
                        print("Queue is full")
                    
                    # Calculate and display FPS every second
                    frame_count += 1
                    if frame_count % 30 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        frame_count = 0
                        start_time = end_time
                    
                    # Add FPS text to the image
                    cv2.putText(image_data, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # Display the image using OpenCV (much faster than matplotlib)
                    cv2.imshow(window_name, image_data)
                    
                    # Process any OpenCV GUI events (like window closing)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break

                    
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


def track(data_queue1:queue.Queue,data_queue2:queue.Queue,stream=True):
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
    point = [0,0,0,0,0,0,0,0]
    fps = 0
    old_time = time.time()

    while True:
        fps = time.time() - old_time
        old_time = time.time()
        fps = 1 / fps if fps > 0 else 0
        try:
            if not (data_queue1.empty() or data_queue2.empty()):
                data1 = data_queue1.get_nowait()
                data2 = data_queue2.get_nowait()
                image_points = [data1,data2]
                object_points,image_p = find_point_correspondance_and_object_points(image_points,camera_poses,4)
                if stream:
                    if len(object_points) > 0:
                        point = object_points[0]
                        point = list(point)
                        point = [0,0,0,0] + point
                    data = {"tracker1": point}
                    connection.send(msgpack.packb(data, use_bin_type=True))
                    print(f"Object Points: {point}")
                else:
                    print(f"Object Points: {object_points}")
                print(f"Image Points: {image_p}")
                print(f"FPS: {fps:.2f}")
                # print(f"Data1: {data1}")
                # print(f"Data2: {data2}")
        except queue.Empty:
            print("Queue is empty")
        except ConnectionResetError:
            while True:
                    print("\nUnity disconnected, waiting for reconnection...")
                    connection, _ = server.accept()
                    print("Connected!")
                    break
            
        time.sleep(0.01)
        #send 3d points

def run_single_camera(cam,data_queue):
    """
    Camera initialization and execution function.
    
    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Initialize camera
        cam.Init()
        
        # Retrieve nodemap
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        nodemap = cam.GetNodeMap()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        exposure_time = 5000  # 1 ms
        cam.ExposureTime.SetValue(exposure_time)
        # Performance optimization: Set packet size to max for GigE cameras
        try:
            # Check if this is a GigE camera
            node_device_type = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode('DeviceType'))
            if (PySpin.IsReadable(node_device_type) and 
                node_device_type.GetCurrentEntry().GetSymbolic() == 'GigEVision'):
                
                # Get GigE specific nodemap
                nodemap_gige = cam.GetTLStreamNodeMap()
                
                # Set stream packet resend
                node_packet_resend = PySpin.CBooleanPtr(nodemap_gige.GetNode('StreamPacketResendEnable'))
                if PySpin.IsWritable(node_packet_resend):
                    node_packet_resend.SetValue(True)
                    print('Packet resend enabled')
                    
                # Set packet size to max
                node_packet_size = PySpin.CIntegerPtr(nodemap_gige.GetNode('StreamPacketSize'))
                if PySpin.IsWritable(node_packet_size):
                    max_packet_size = node_packet_size.GetMax()
                    node_packet_size.SetValue(max_packet_size)
                    print(f'Packet size set to maximum ({max_packet_size})')

                
                # binning_horizontal = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
                # if PySpin.IsAvailable(binning_horizontal) and PySpin.IsWritable(binning_horizontal):
                #     binning_horizontal.SetValue(1224)  # 2x binning

                
                # binning_vertical = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
                # if PySpin.IsAvailable(binning_vertical) and PySpin.IsWritable(binning_vertical):
                #     binning_vertical.SetValue(1024)  # 2x binning
        except PySpin.SpinnakerException as ex:
            print(f'Notice: GigE optimization not applicable - {ex}')
            
        # Run acquisition and display function
        result = track_points(cam, nodemap, nodemap_tldevice,data_queue=data_queue,preview=False)
        
        # Deinitialize camera
        cam.DeInit()
        return result
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False


def main():
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
        
        data_queue1 = queue.Queue(maxsize=10)
        data_queue2 = queue.Queue(maxsize=10)

        process_thread = threading.Thread(target=track, args=(data_queue1,data_queue2,))
        process_thread.start()        
        camera1_display = threading.Thread(target=run_single_camera, args=(cam_list[0],data_queue1))
        camera1_display.start()
        camera2_display = threading.Thread(target=run_single_camera, args=(cam_list[1],data_queue2))
        camera2_display.start()

        while running.is_set():
            time.sleep(0.1)
            # print('Running...')
        
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

#else
#ifdef Azure

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

#else
#ifdef Unity

# Unity camera configuration (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

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
            PORT = 5002  # Different port to avoid conflict with IMU
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
#endifall

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
