import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv
import numpy as np
from lib.Helpers import get_extrinsics, triangulate_points, calculate_reprojection_errors, bundle_adjustment,get_extrinsics,find_point_correspondance_and_object_points
from CapturePoints_macro_template import read_images, capture_floor_points
import matplotlib.pyplot as plt
import json
from itertools import combinations

images = []
objs = []
R = None
origin = None
objs = []
R = None
origin = None
global_camera_poses = None
camera_count = 2

intrinsics_json = "./jsons/camera-params-in.json"
points_json = "./jsons/image_points.json"

camera_params_file = open(intrinsics_json)
camera_params = json.load(camera_params_file)
# camera_poses, camera_count = get_extrinsics()

def poses_to_fundamental_matrix(pose1, pose2, K1=None, K2=None):
    """
    Convert two camera poses to fundamental matrix in OpenCV format.
    
    Args:
        pose1: Dictionary with 'R' (3x3 rotation matrix) and 't' (3x1 translation vector) for camera 1
        pose2: Dictionary with 'R' (3x3 rotation matrix) and 't' (3x1 translation vector) for camera 2
        K1: Camera intrinsic matrix for camera 1 (3x3). If None, assumes identity (normalized coordinates)
        K2: Camera intrinsic matrix for camera 2 (3x3). If None, assumes identity (normalized coordinates)
    
    Returns:
        F: Fundamental matrix (3x3) compatible with OpenCV format
        
    Note:
        - If K1 and K2 are provided, F relates pixel coordinates
        - If K1 and K2 are None, F relates normalized image coordinates
        - The fundamental matrix satisfies: x2^T * F * x1 = 0 for corresponding points
    """
    
    # Extract rotation and translation
    R1, t1 = pose1['R'], pose1['t'].reshape(-1, 1)
    R2, t2 = pose2['R'], pose2['t'].reshape(-1, 1)
    
    # Compute relative pose from camera 1 to camera 2
    # Camera 1 coordinate system -> World -> Camera 2 coordinate system
    R_rel = R2 @ R1.T  # Relative rotation
    t_rel = t2 - R2 @ R1.T @ t1  # Relative translation
    
    # Create skew-symmetric matrix from translation vector
    def skew_symmetric(v):
        """Create skew-symmetric matrix from 3D vector"""
        v = v.flatten()
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    # Essential matrix E = [t]_× * R
    t_skew = skew_symmetric(t_rel)
    E = t_skew @ R_rel
    
    # If intrinsic matrices are provided, compute fundamental matrix
    # F = K2^(-T) * E * K1^(-1)
    if K1 is not None and K2 is not None:
        K1_inv = np.linalg.inv(K1)
        K2_inv_T = np.linalg.inv(K2).T
        F = K2_inv_T @ E @ K1_inv
    else:
        # For normalized coordinates, F = E
        F = E
    
    return F

def get_points(path=None):
    '''Read image points from json'''
    global points_json
    if path is not None:
        points_json = path
    with open(points_json) as file:
        image_points = json.load(file)
    image_points = np.array(image_points)
    image_points = np.transpose(image_points, (1, 0, 2))
    return image_points

def draw_epipolar_lines(img1, img2, points1, points2, F,camera_params):
    """
    Draw epipolar lines on two images based on corresponding points.
    
    Args:
        img1, img2: Input images
        points1, points2: Corresponding points in the images
        F: Fundamental matrix
    """
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)
    
    img1_lines = img1.copy()
    img2_lines = img2.copy()
    img1_lines = cv.undistort(img1_lines, np.array(camera_params[0]["intrinsic_matrix"]), np.array(camera_params[0]["distortion_coef"]))
    img2_lines = cv.undistort(img2_lines, np.array(camera_params[1]["intrinsic_matrix"]), np.array(camera_params[1]["distortion_coef"]))
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    colors = np.random.randint(0, 255, (len(points1), 3)).tolist()
    
    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    for i, (pt, line) in enumerate(zip(points1, lines2)):
        color = colors[i]
        
        # Draw point in first image
        x, y = map(int, pt)
        cv.circle(img1_lines, (x, y), 10, color, -1)
        
        # Draw corresponding epipolar line in second image
        a, b, c = line
        x0, y0 = 0, int(-c/b) if b != 0 else 0
        x1, y1 = w2, int(-(a*w2+c)/b) if b != 0 else 0
        cv.line(img2_lines, (x0, y0), (x1, y1), color, 3)
        
        # Draw corresponding point in second image
        if i < len(points2):
            x, y = map(int, points2[i])
            cv.circle(img2_lines, (x, y), 10, color, -1)
    
    # Draw lines in first image
    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    for i, line in enumerate(lines1):
        color = colors[i]
        
        # Draw epipolar line in first image
        a, b, c = line
        x0, y0 = 0, int(-c/b) if b != 0 else 0
        x1, y1 = w1, int(-(a*w1+c)/b) if b != 0 else 0
        cv.line(img1_lines, (x0, y0), (x1, y1), color, 3)
    
    img1_rgb = cv.cvtColor(img1_lines, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2_lines, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121), plt.imshow(img1_rgb)
    plt.title('Image 1 with epipolar lines')
    plt.axis('off')
    
    plt.subplot(122), plt.imshow(img2_rgb)
    plt.title('Image 2 with epipolar lines')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_extrinsics(image_points):
    '''Calculate extrinsics from image points'''
    global camera_params
    global global_camera_poses
    
    camera_count = len(image_points)
    
    Fs = []

    global_camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]

    image1 = cv.imread("macro_for_cam_SDKs/Kinect_Azure_x_Kinect_OLD/tracking/cam0/1766123505_color_azure.png")
    image2 = cv.imread("macro_for_cam_SDKs/Kinect_Azure_x_Kinect_OLD/tracking/cam1/1766123505_color_kv2.png")

    camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]
    for camera_i in range(0, camera_count-1):
        camera1_image_points = image_points[camera_i]
        camera2_image_points = image_points[camera_i+1]
        camera1_image_points = np.array(camera1_image_points, dtype=np.float32)
        camera2_image_points = np.array(camera2_image_points, dtype=np.float32)

        # Add validation for sufficient points
        if len(camera1_image_points) < 8 or len(camera2_image_points) < 8:
            print(f"Insufficient points for fundamental matrix calculation. Need at least 8 points, got {len(camera1_image_points)} and {len(camera2_image_points)}")
            continue

        print(f"Camera1 points: {camera1_image_points}")
        print(f"Camera2 points: {camera2_image_points}")

        F, mask = cv.findFundamentalMat(camera1_image_points, camera2_image_points, cv.FM_RANSAC, 1.0, 0.99)
        
        # Add check for None fundamental matrix
        if F is None:
            print("Failed to compute fundamental matrix. This could be due to:")
            print("1. Insufficient corresponding points")
            print("2. Points are coplanar")
            print("3. Poor point quality or matching")
            print("Try capturing more diverse point correspondences.")
            continue
        
        print(f"F: {F}")
        Fs.append(F.tolist())

        K1 = np.array(camera_params[0]["intrinsic_matrix"])
        K2 = np.array(camera_params[1]["intrinsic_matrix"])

        E = K2.T @ F @ K1

        R1, R2, t = cv.decomposeEssentialMat(E)

        draw_epipolar_lines(image1, image2, camera1_image_points, camera2_image_points, F,camera_params)

        possible_Rs = [R1, R1, R2, R2]
        possible_ts = [t, -t, t, -t]

        R = None
        t = None
        max_points_infront_of_camera = 0
        for i in range(0, 4):
            
            print("possible",possible_Rs[i],possible_ts[i])
            p = [camera1_image_points,camera2_image_points]
            p = np.transpose(p,[1,0,2])
            # print(p.shape)
            object_points = triangulate_points(p, np.concatenate([[camera_poses[-1]], [{"R": possible_Rs[i], "t": possible_ts[i]}]]))
            object_points_camera_coordinate_frame = np.array([possible_Rs[i].T @ object_point for object_point in object_points])
            couples = zip(range(1,len(camera1_image_points)+1),camera1_image_points,object_points,camera2_image_points, object_points_camera_coordinate_frame)
            for point in couples:
                print(f"object point {point[0]}: ", point[1], point[2], point[3],point[4])

            points_infront_of_camera = np.sum(object_points[:,2] > 0) + np.sum(object_points_camera_coordinate_frame[:,2] > 0)

            if points_infront_of_camera > max_points_infront_of_camera:
                max_points_infront_of_camera = points_infront_of_camera
                R = possible_Rs[i]
                t = possible_ts[i]

        R = R @ camera_poses[-1]["R"]
        t = camera_poses[-1]["t"] + (camera_poses[-1]["R"] @ t)
        print("R:", R)
        print("t:", t)
        camera_poses.append({
            "R": R,
            "t": t
        })
    
    with open("./jsons/fundamentals.json", "w") as outfile:
        json.dump(Fs, outfile)
    
    with open("./jsons/fundamentals.json", "w") as outfile:
        json.dump(Fs, outfile)

    global_camera_poses = camera_poses
    save_extrinsics("before_ba_")
    image_points = np.transpose(image_points, (1, 0, 2))
    camera_poses = bundle_adjustment(image_points, camera_poses)
    print("Camera poses after BA: ", camera_poses)
    F_new = poses_to_fundamental_matrix(camera_poses[0], camera_poses[1], camera_params[0]["intrinsic_matrix"], camera_params[1]["intrinsic_matrix"])
    draw_epipolar_lines(image1, image2, camera1_image_points, camera2_image_points, F_new,camera_params)
    object_points = triangulate_points(image_points, camera_poses)
    save_objects("after_ba_",object_points)
    error = np.mean(calculate_reprojection_errors(image_points, object_points, camera_poses))
    global_camera_poses = camera_poses
    print("Reprojection error:", error)
    save_extrinsics("after_ba_")
    save_objects(prefix="after_ba_",object_points=object_points)

def save_extrinsics(prefix=""):
    '''Save camera poses to a json'''
    global global_camera_poses
    global camera_count

    extrinsics = []
    for i in range(0, camera_count):
        extrinsics.append({
            "R": global_camera_poses[i]["R"].tolist(),
            "t": global_camera_poses[i]["t"].flatten().tolist()
        })

    extrinsics_filename = f"./jsons/{prefix}extrinsics.json"
    with open(extrinsics_filename, "w") as outfile:
        json.dump(extrinsics, outfile)

    print("Extrinsics saved to", extrinsics_filename)

def save_objects(prefix="",object_points=None):
    '''Save object (which were used to calculate the camera poses) to a json'''
    objects_filename = f"./jsons/{prefix}objects.json"
    with open(objects_filename, "w") as outfile:
        json.dump(object_points.tolist(), outfile)

    print("Object points saved to", objects_filename)

def set_origin(points_3d):
    global global_camera_poses
    global origin
    print("prev",global_camera_poses)
    if len(points_3d) > 2:
        origin = np.mean(points_3d, axis=0)
        print("Origin:", origin)
        for pose in global_camera_poses:
            pose['t'] = pose['t'] - origin
    else:
        print("Invalid number of points to calculate origin")
    print("after",global_camera_poses)

def set_floor(points_3d):
    global global_camera_poses
    positions = list(range(0, len(points_3d)))
    coms = list(combinations(positions, 3))
    if len(points_3d) > 2:
        normals = []
        for combination in coms:
            normal = calculate_normal([points_3d[combination[0]], points_3d[combination[1]], points_3d[combination[2]]])
            normals.append(normal)
        normal = np.mean(normals, axis=0)
        print("Normal: ",normal)
        global R
        R = rotation_matrix_from_vectors(np.array([0, 0, -1]), normal) # [0,0,-1] can be changed to [0,0,1] if the normal is pointing in the wrong direction
        R = np.pad(R, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        R[3, 3] = 1
        print("poses: ", global_camera_poses[1])
        for i in range(len(global_camera_poses)):
            RT = np.eye(4)
            RT[:3,:3] = global_camera_poses[i]['R']
            RT[:3,3] = global_camera_poses[i]['t'].flatten()
            RT = R.T @ RT # idk why this works but it does should it be RT = RT @ R.T?
            global_camera_poses[i]['R'] = RT[:3,:3]
            global_camera_poses[i]['t'] = RT[:3,3].reshape(3,1)
        print("poses: ", global_camera_poses[1])
    else:
        print("Invalid number of points to calculate floor")
    save_extrinsics("after_floor_")

def calculate_normal(points_3d):
    if len(points_3d) == 3:
        normal = np.cross(points_3d[1]-points_3d[0], points_3d[2]-points_3d[0])
        normal = 0 if np.linalg.norm(normal) == 0 else normal/np.linalg.norm(normal)
        return normal
    else:
        print("Invalid number of points to calculate normal")
        return np.array([0,0,1])

def rotation_matrix_from_vectors(vec_orig, vec_rot):

    vec_orig = vec_orig / np.linalg.norm(vec_orig)
    vec_rot = vec_rot / np.linalg.norm(vec_rot)

    cross_prod = np.cross(vec_orig, vec_rot)
    cross_norm = np.linalg.norm(cross_prod)
    
    dot_prod = np.dot(vec_orig, vec_rot)
    
    if cross_norm == 0:
        if dot_prod > 0:
            return np.eye(3)
        else:
            axis = np.array([1, 0, 0]) if abs(vec_orig[0]) < 0.99 else np.array([0, 1, 0])
            cross_prod = np.cross(vec_orig, axis)
            cross_prod /= np.linalg.norm(cross_prod)
            cross_norm = 1

    K = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])

    I = np.eye(3)
    R = I + K + K @ K * ((1 - dot_prod) / (cross_norm ** 2))
    
    return np.array(R)

def origin_and_floor():
    global global_camera_poses
    global camera_count
    global_camera_poses, camera_count = get_extrinsics()
    # print("Camera poses: ", global_camera_poses)
    images = read_images(path="macro_for_cam_SDKs/floor_images",debug=True)
    points = capture_floor_points(preview=True,images=images)
    print("Points:", points)
    
    # Check if we have valid points
    valid_cameras_count = 0
    total_valid_points = 0
    
    for i, camera_points in enumerate(points):
        valid_points_in_camera = 0
        for point in camera_points:
            if point[0] is not None and point[1] is not None:
                valid_points_in_camera += 1
                total_valid_points += 1
        
        if valid_points_in_camera > 0:
            valid_cameras_count += 1
            print(f"Camera {i} has {valid_points_in_camera} valid points")
    
    if valid_cameras_count == 0:
        print("No valid points detected in any camera. Cannot proceed with origin and floor calculation.")
        return
    elif valid_cameras_count == 1:
        print(f"Points detected in only one camera. Need at least 2 cameras for triangulation.")
        print("Possible solutions:")
        print("1. Check the floor image for the other camera")
        print("2. Ensure the ping-pong balls are visible in both cameras")
        print("3. Adjust lighting or camera positions")
        return
    
    print(f"Valid cameras: {valid_cameras_count}, Total valid points: {total_valid_points}")
    
    # For floor calculation, we typically expect 4 corner points
    if total_valid_points < 4:
        print(f"Expected at least 4 points for floor calculation, but got {total_valid_points} valid points")
        print("You may proceed with fewer points, but accuracy will be reduced.")
        # Uncomment the return below if you want to enforce 4 points
        # return
    
    try:
        object_points, image_points_coupled = find_point_correspondance_and_object_points(points, global_camera_poses, total_valid_points)
        print("Object points: ", object_points)
        set_origin(object_points)
        save_extrinsics("after_origin_")
        #set_floor(object_points)
    except Exception as e:
        print(f"Error in triangulation: {e}")
        print("This could be due to:")
        print("1. Insufficient point correspondences between cameras")
        print("2. Poor camera calibration")
        print("3. Points not visible in both cameras")

if __name__ == "__main__":
    image_points = get_points()
    camera_poses =  calculate_extrinsics(image_points)
    origin_and_floor()
