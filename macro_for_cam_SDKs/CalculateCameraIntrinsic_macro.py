import cv2
import numpy as np
import glob
import json
import sys

# ---- Configuration (kept single JSON path/schema) ----
CHECKERBOARD = (9, 6)       # inner corners (cols, rows)
SQUARE_SIZE = 0.025  # in meters
CAM_IMAGES_FOLDER = "checkerboard"
JSON_PATH = "./jsons/camera-intrinsics.json"
IMAGE_EXTS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")

# Common inner-corner sizes to try (we’ll also try the swapped orientation)
ALT_CHECKERBOARD_SIZES = [(9,6), (8,6), (7,5), (10,7), (11,8)]


def _prepare_objp(board, square_size):
    cols, rows = board
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)
    return objp


def _gray8_from_img(img):
    """Return enhanced 8-bit grayscale from BGR or 8-bit grayscale."""
    if img is None:
        return None
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img

    g = g.astype(np.uint8, copy=False)

    # mild highlight clipping and normalization to help checker detection
    p_hi = np.percentile(g, 99.0)
    g = np.minimum(g, int(p_hi)).astype(np.uint8)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    # local contrast (CLAHE) to sharpen edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    return g


def _normalize_corners(c):
    if c is None or isinstance(c, (bool, np.bool_)):
        return None
    if isinstance(c, tuple):
        c = c[1] if len(c) >= 2 else (None if len(c) == 0 else c[0])
    a = np.asarray(c)
    if a.ndim == 0:
        return None
    if a.ndim == 2 and a.shape[1] == 2:
        a = a.reshape(-1, 1, 2)
    elif not (a.ndim == 3 and a.shape[1] == 1 and a.shape[2] == 2):
        try:
            a = a.reshape(-1, 1, 2)
        except Exception:
            return None
    return a.astype(np.float32)


def _find_corners(gray, board):
    """Try SB first, then classic; also try inverted image."""
    flags_classic = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FILTER_QUADS)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    target_n = board[0] * board[1]

    for img_try in (gray, 255 - gray):
        # SB (if available)
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                c = _normalize_corners(cv2.findChessboardCornersSB(img_try, board, flags=flags_sb))
                if c is not None and c.shape[0] == target_n:
                    return True, c, board
            except Exception:
                pass
        # classic
        ret, c = cv2.findChessboardCorners(img_try, board, flags=flags_classic)
        c = _normalize_corners(c) if ret else None
        if c is not None and c.shape[0] == target_n:
            return True, c, board
    return False, None, board

def _try_detect_one(gray, boards):
    """Detect corners over several boards and both polarities."""
    for img_try in (gray, 255 - gray):
        for b in boards:
            ok, corners, used = _find_corners(img_try, b)
            if ok:
                return True, corners, used
    return False, None, None

def _detect_corners_in_images(image_paths, board, square_size, wait_ms):
    objpoints, imgpoints = [], []
    image_size = None

    if not image_paths:
        return objpoints, imgpoints, image_size

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    win = "Checkerboard"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Build candidate boards (include swapped orientation)
    boards = []
    for b in ALT_CHECKERBOARD_SIZES:
        boards.extend([b, (b[1], b[0])])

    for f in image_paths:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = _gray8_from_img(img)
        image_size = gray.shape[::-1]

        ok, corners, used = _try_detect_one(gray, boards)
        if not ok:
            # Try gentle upscale if the pattern is small
            gray_up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            ok, corners, used = _try_detect_one(gray_up, boards)
            if ok:
                corners = corners / 1.5  # map back to original size

        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if ok:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            cv2.drawChessboardCorners(disp, used, corners2, True)
            objpoints.append(_prepare_objp(used, square_size))
            imgpoints.append(corners2)
            print(f"Detected board {used} in {f}")
        else:
            print(f"No board found in {f}")

        if (cv2.waitKey(wait_ms) & 0xFF) == ord('q'):
            break

        cv2.imshow(win, disp)

    cv2.destroyWindow(win)
    return objpoints, imgpoints, image_size


def _calibrate(objpoints, imgpoints, image_size):
    if not objpoints or not imgpoints or image_size is None:
        return None, None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # reprojection RMSE
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_pts += n
    mean_rmse = float(np.sqrt(total_err / max(total_pts, 1)))

    return ret, mtx, dist, rvecs, tvecs, mean_rmse


def calculate_camera_intrinsics(wait_time=1):
    """
    Calculate camera intrinsics using checkerboard images.
    Returns tuple: (camera matrix, distortion coefficients)
    """
    # Collect images with multiple extensions
    image_names = []
    for ext in IMAGE_EXTS:
        image_names.extend(glob.glob(f'macro_for_cam_SDKs/{CAM_IMAGES_FOLDER}/*.{ext}'))
    image_names = sorted(image_names)
    if not image_names:
        print(f"Error: No images found in 'macro_for_cam_SDKs/{CAM_IMAGES_FOLDER}'!")
        return None, None

    objp, imgp, size = _detect_corners_in_images(image_names, CHECKERBOARD, SQUARE_SIZE, wait_time)
    if not objp or not imgp or size is None:
        print("Error: No checkerboard patterns found in images!")
        print("Make sure you have checkerboard images in the 'checkerboard' folder")
        return None, None

    ret, mtx, dist, rvecs, tvecs, rmse = _calibrate(objp, imgp, size)
    if not ret:
        print("Camera calibration failed!")
        return None, None

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.reshape(-1))
    print(f"Reprojection RMSE: {rmse:.4f}")
    return mtx, dist


def save_intrinsics(mtx, dist):
    """
    Save the camera intrinsics to a JSON file.
    Schema kept same as original macro.
    """
    mtx = mtx.tolist()
    dist = dist.tolist()
    intrinsics = {"intrinsic_matrix": mtx, "distortion_coef": dist[0]}
    intrinsic_filename = JSON_PATH

    try:
        with open(intrinsic_filename, "w") as outfile:
            json.dump(intrinsics, outfile)
        return True
    except Exception as e:
        print(f"Error saving intrinsics: {e}")
        return False


if __name__ == "__main__":
    result = calculate_camera_intrinsics(1)
    if result is not None:
        mtx, dist = result
        if mtx is not None and dist is not None:
            save_intrinsics(mtx, dist)
        else:
            print("Failed to calculate camera intrinsics")
    else:
        print("No calibration data available")
