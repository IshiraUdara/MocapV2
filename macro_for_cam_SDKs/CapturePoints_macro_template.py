import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv
import numpy as np
import os
import glob
from lib.ImageOperations import _find_dot
from lib.Helpers import get_extrinsics
import json

points_json = "./jsons/image_points.json"

intrinsics_json = "./jsons/camera-params-in.json"
camera_params_file = open(intrinsics_json)
camera_params = json.load(camera_params_file)

camera_poses, _ = get_extrinsics()


def read_images(path,preview=False,debug=False):
    '''Read images in folders
    
    Parameters:
        preview (bool): If True, show the images in a window.
        debug (bool): If True, print debug information.

    Returns:
        images (numpy array): [[camera1_images], [camera2_images], ...]
    '''
    images = []
    print(path)
    img_folder_paths = sorted([f"{path}/{i}" for i in os.listdir(f"{path}") if i.startswith('cam')])
    camera_count = len(img_folder_paths)
    if debug:
        print(img_folder_paths)

    window = cv.namedWindow(f'Preview', cv.WINDOW_NORMAL)
    for img_folder_path in img_folder_paths:
        image_names = sorted(glob.glob(f'./{img_folder_path}/*.png'))
        image_count = len(image_names)
        if debug:
            print(image_names)
        imgs = []
        for fname in image_names:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            if preview:
                cv.imshow(f'Preview',gray)
                key = cv.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    cv.destroyAllWindows()
                    quit()
            imgs.append(gray)
        images.append(imgs)

    images = np.array(images)
    if debug:
        print("Point count:", image_count)
        print("Camera count:", camera_count)
        print("Images shape",images.shape)

    return images



def capture_pose_points(images,preview=False,debug=False):
    '''output: saves points to points_json
    prerequisites needed: images shape: (camera_count, image_count, height, width, channels)
    
    Parameters:
        images (numpy array) : Images for callibration in the shape of (camera_count, image_count, height, width, channels)
        preview (bool) : If True, show the images in a window.
        debug (bool) : If True, print details in the terminal.

    Returns:
        image_points (numpy array) : [[camera1_points], [camera2_points], ...] -> camera1_points = [timestamp1, timestamp2, ...]
    '''


    image_points = []
    camera_count = len(images)
    image_count = len(images[0])
    cv.namedWindow(f'Preview', cv.WINDOW_NORMAL)

    # Global variables for mouse callback function
    clicked_point = None
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_point
        if event == cv.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            print(f"Clicked point: {clicked_point}")
    
    # Set the mouse callback function for the window
    cv.setMouseCallback('Preview', mouse_callback)

    for j in range(0, image_count):
        processed_images = []
        calculated_points = np.zeros((camera_count, 2))
        skip = False
        
        for i in range(0, camera_count):
            image, detected_points = _find_dot(images[i][j])
            
            if not detected_points == [[None, None]]:
                # Display image with detected points
                display_image = image.copy()

                # Clear prev clicked point
                clicked_point = None

                
                while clicked_point is None:
                    cv.imshow('Preview', display_image)
                    cv.putText(display_image, "Click on the desired point", (10, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    key = cv.waitKey(10)
                    if key == 27:  # ESC key to exit
                        cv.destroyAllWindows()
                        quit()
                # Find the closest detected point to where the user clicked
                min_dist = float('inf')
                closest_point = None
                idx = -1

                for k in range(len(detected_points)):
                    dist = ((detected_points[k][0] - clicked_point[0]) ** 2 + (detected_points[k][1] - clicked_point[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = detected_points[k]
                        idx = k
                
                if closest_point is None:
                    print("No points detected near click. Please try again.")
                    i -= 1  # Retry this camera
                    continue
                    
                # Use the closest detected point
                image_point = np.array(closest_point)
            else:
                print("No points detected.Skipping the set.")
                image_point = [None, None]
                skip = True
                break
            print(f"{idx} is clicked!")
            calculated_points[i] = image_point
            processed_images.append(image)
            if len(image_point) != 2:
                print("Found more than one point or no point in the image . Please make sure there is only one point in the image.")
                quit()
            else:
                image_point = image_point[0]
        if not skip:
            if preview:
                image = cv.resize(image, (int(image.shape[1] * 0.25), int(image.shape[0] * 0.25)))
                image = np.hstack([processed_images[0], processed_images[1]])
                cv.imshow("Preview", image)
                key = cv.waitKey(0) & 0xFF
                if key == ord(' '):
                    print("Saving points...")
                    image_points.append(calculated_points.tolist())
                if key == ord('x'):
                    print("Skipping points...")
                if key == ord('q'):
                    print("Exiting...")
                    cv.destroyAllWindows()
                    quit()
            else:
                image_points.append(calculated_points.tolist())
            
            if debug:
                print(f"{len(image_points)}. Image points for timestamp", j+1, ":", calculated_points) 

    cv.destroyAllWindows()
    return image_points


def save_points(image_points):
    """ Save image points to a json"""
    global points_json
    with open(points_json, "w") as file:
        json.dump(image_points, file)


def capture_floor_points(preview=False,debug=False,images=None,camera_count=2):
    '''output: calculated points
    prerequisites needed: images shape: (camera_count, 1, height, width, channels)'''
    if preview:
        print("Press 'space' to take a picture and 'q' to quit.")

    processed_images = []
    calculated_points = []
    
    for i in range(0, camera_count):
        image, image_points = _find_dot(images[i][0], print_location=True)
        
        # Handle the case where no points are detected
        if image_points == [[None, None]] or not image_points or image_points is None:
            print(f"No points detected in camera {i}")
            # Create consistent structure for no points
            calculated_points.append([[None, None]])
        else:
            # Ensure image_points is in the correct format
            if isinstance(image_points, list) and len(image_points) > 0:
                # Convert to consistent format - list of [x, y] pairs
                formatted_points = []
                for point in image_points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        formatted_points.append([float(point[0]), float(point[1])])
                    elif hasattr(point, '__len__') and len(point) >= 2:
                        formatted_points.append([float(point[0]), float(point[1])])
                
                if formatted_points:
                    calculated_points.append(formatted_points)
                else:
                    print(f"Invalid point format in camera {i}")
                    calculated_points.append([[None, None]])
            else:
                print(f"Unexpected image_points format in camera {i}: {image_points}")
                calculated_points.append([[None, None]])
        
        processed_images.append(image)
    
    cv.namedWindow(f'Preview', cv.WINDOW_NORMAL)
    if preview:
        if len(processed_images) >= 2:
            # Resize images if they're different sizes
            h1, w1 = processed_images[0].shape[:2]
            h2, w2 = processed_images[1].shape[:2]
            if h1 != h2 or w1 != w2:
                # Resize second image to match first
                processed_images[1] = cv.resize(processed_images[1], (w1, h1))
            image = np.vstack([processed_images[0], processed_images[1]])
        else:
            image = processed_images[0]
        cv.imshow("Preview", image)
        key = cv.waitKey(0) & 0xFF
        if key == ord(' '):
            print("Getting points...")
        if key == ord('q'):
            print("Exiting...")
            cv.destroyAllWindows()
            quit()
    
    if debug:
        print("All calculated points:", calculated_points)
        for i, points in enumerate(calculated_points):
            print(f"Image points for camera {i}: {points}")

    cv.destroyAllWindows()
    
    # Handle the mixed results case
    # Check if we have at least one camera with valid points
    valid_cameras = []
    for i, camera_points in enumerate(calculated_points):
        if camera_points and camera_points != [[None, None]]:
            # Check if any point in this camera is valid
            has_valid_points = any(p[0] is not None and p[1] is not None for p in camera_points)
            if has_valid_points:
                valid_cameras.append(i)
    
    if len(valid_cameras) == 0:
        print("No valid points detected in any camera")
        return np.array([[[None, None]] for _ in range(camera_count)])
    
    # For floor point calculation, we might need manual intervention if only one camera has points
    if len(valid_cameras) == 1:
        print(f"Only camera {valid_cameras[0]} has detected points: {calculated_points[valid_cameras[0]]}")
        print("You may need to:")
        print("1. Manually mark points in the other camera")
        print("2. Check lighting/visibility conditions")
        print("3. Adjust detection parameters")
        
        # For now, return the structure with None for the camera without points
        result = []
        for i in range(camera_count):
            if i in valid_cameras:
                result.append(calculated_points[i])
            else:
                result.append([[None, None]])
        
        try:
            return np.array(result, dtype=object)  # Use object dtype to handle mixed None/float arrays
        except Exception as e:
            print(f"Error creating numpy array: {e}")
            return result  # Return as list if numpy array creation fails
    
    # If we have points from multiple cameras, proceed normally
    try:
        # Ensure all cameras have same number of points (pad with None if needed)
        max_points = max(len(points) for points in calculated_points if points != [[None, None]])
        normalized_points = []
        
        for camera_points in calculated_points:
            if camera_points == [[None, None]]:
                normalized_points.append([[None, None]] * max_points)
            else:
                # Pad or truncate to match max_points
                while len(camera_points) < max_points:
                    camera_points.append([None, None])
                normalized_points.append(camera_points[:max_points])
        
        return np.array(normalized_points, dtype=object)
    except Exception as e:
        print(f"Error converting to numpy array: {e}")
        print(f"Calculated points structure: {calculated_points}")
        return calculated_points  # Return as list if numpy array creation fails


if __name__ == "__main__":
    images = read_images(path="macro_for_cam_SDKs/Kinect_x_EZVIZ/captured_images",debug=True)
    points = capture_pose_points(images,preview=True,debug=True)
    print(points)
    save_points(points)