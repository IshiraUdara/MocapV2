import os
import time
import json
import glob
import cv2
import numpy as np

# Optional device SDKs
try:
    import pykinect_azure as pykinect
    from pykinect_azure.k4a import Image, _k4a
except Exception:
    pykinect = None
    Image = None
    _k4a = None

try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
except Exception:
    PyKinectRuntime = None
    PyKinectV2 = None

# ---- Configuration ----
CHECKERBOARD = (9, 6)         # inner corners (cols, rows)
SQUARE_SIZE = 0.025           # unitless for intrinsics; set actual size if you want (e.g., meters)
MIN_SAMPLES = 12              # minimum successful detections required
WAIT_MS = 1                   # cv2.waitKey delay

# Folders for pre-captured IR images
AZURE_IR_GLOB = 'Kinect_DK_cam/captured_images/cam0/*.png'
KV2_IR_GLOB   = 'Kinect_DK_cam/captured_images/cam1/*.png'

JSON_DIR = './jsons'
AZURE_JSON = os.path.join(JSON_DIR, 'azure_ir_intrinsics.json')
KV2_JSON   = os.path.join(JSON_DIR, 'kinectv2_ir_intrinsics.json')


def _prepare_objp(board, square_size):
    cols, rows = board
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)
    return objp


def _gray8_from_ir(img):
    """Return 8-bit grayscale from 8/16-bit IR or BGR, with IR-friendly enhancement."""
    if img is None:
        return None
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    # convert to 8-bit safely
    if g.dtype == np.uint16:
        g = cv2.convertScaleAbs(g, alpha=255.0/65535.0)
    else:
        g = g.astype(np.uint8, copy=False)
    # clip top highlights (suppress saturated IR glare) and normalize
    p_hi = np.percentile(g, 99.0)
    g = np.minimum(g, int(p_hi)).astype(np.uint8)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    # local contrast for edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    return g


def _find_corners(gray, board):
    """Try SB first, then classic; also try inverted image. Returns (found, corners, used_board)."""
    def normalize(c):
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

    flags_classic = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FILTER_QUADS)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    target_n = board[0] * board[1]

    for img_try in (gray, 255 - gray):
        # SB
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                c = normalize(cv2.findChessboardCornersSB(img_try, board, flags=flags_sb))
                if c is not None and c.shape[0] == target_n:
                    return True, c, board
            except Exception:
                pass
        # classic
        ret, c = cv2.findChessboardCorners(img_try, board, flags=flags_classic)
        c = normalize(c) if ret else None
        if c is not None and c.shape[0] == target_n:
            return True, c, board
    return False, None, board


def _detect_corners_in_images(image_paths, board, square_size):
    objpoints = []
    imgpoints = []
    gray_shape = None

    if not image_paths:
        return objpoints, imgpoints, gray_shape

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.namedWindow("IR Checkerboard (images)", cv2.WINDOW_NORMAL)

    # try both orientations
    candidates = [board, (board[1], board[0])]

    for f in image_paths:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = _gray8_from_ir(img)
        gray_shape = gray.shape[::-1]

        found = False
        for b in candidates:
            ok, corners, used = _find_corners(gray, b)
            if ok:
                objp = _prepare_objp(used, square_size)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(disp, used, corners2, True)
                found = True
                break

        if not found:
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imshow("IR Checkerboard (images)", disp)
        if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("IR Checkerboard (images)")
    return objpoints, imgpoints, gray_shape


def _collect_live_azure(board, square_size, min_samples=MIN_SAMPLES):
    if pykinect is None or _k4a is None:
        print("Azure Kinect SDK not available.")
        return [], [], None

    try:
        pykinect.initialize_libraries(track_body=False)
        cfg = pykinect.default_configuration
        cfg.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        cfg.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        cfg.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        device = pykinect.start_device(config=cfg)
    except Exception as e:
        print(f"Azure init failed: {e}")
        return [], [], None

    objpoints = []
    imgpoints = []
    objp = _prepare_objp(board, square_size)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    win = "Azure IR (live)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    gray_shape = None

    print("Azure IR live capture: show checkerboard; press ESC to stop.")
    try:
        while True:
            capture = device.get_capture()
            if capture is None:
                continue
            ir_handle = _k4a.k4a_capture_get_ir_image(capture.handle)  # fix: pass handle
            if not ir_handle:
                continue
            try:
                w = _k4a.k4a_image_get_width_pixels(ir_handle)
                h = _k4a.k4a_image_get_height_pixels(ir_handle)
                arr = Image(ir_handle).to_numpy()
                ir = arr if not isinstance(arr, tuple) else arr[1]
                if ir is None:
                    continue
                gray = (ir >> 8).astype(np.uint8) if ir.dtype == np.uint16 else ir.astype(np.uint8)
                gray_shape = (w, h)
            finally:
                _k4a.k4a_image_release(ir_handle)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, board, flags)
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                cv2.drawChessboardCorners(disp, board, corners2, True)
                # avoid adding near-duplicate samples by spacing in time
                if len(imgpoints) == 0 or (len(imgpoints) > 0 and time.time() % 0.5 < 0.03):
                    objpoints.append(objp.copy())
                    imgpoints.append(corners2)
                    print(f"Azure detections: {len(imgpoints)}/{min_samples}")

            cv2.imshow(win, disp)
            k = cv2.waitKey(WAIT_MS) & 0xFF
            if k == 27:  # ESC
                break
            if len(imgpoints) >= min_samples:
                break
    finally:
        cv2.destroyWindow(win)
    return objpoints, imgpoints, gray_shape


def _collect_live_kinect_v2(board, square_size, min_samples=MIN_SAMPLES):
    if PyKinectRuntime is None or PyKinectV2 is None:
        print("PyKinect2 not available; Kinect v2 disabled.")
        return [], [], None

    try:
        k2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
    except Exception as e:
        print(f"Kinect v2 init failed: {e}")
        return [], [], None

    objpoints = []
    imgpoints = []
    objp = _prepare_objp(board, square_size)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    win = "Kinect v2 IR (live)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    gray_shape = None

    # Try get default size
    try:
        h, w = k2.infrared_frame_desc.Height, k2.infrared_frame_desc.Width
    except Exception:
        h, w = 424, 512

    print("Kinect v2 IR live capture: show checkerboard; press ESC to stop.")
    try:
        while True:
            frame = k2.get_last_infrared_frame()
            if frame is None:
                continue
            try:
                ir = np.array(frame, dtype=np.uint16).reshape((h, w))
            except Exception:
                ir = frame.reshape((h, w)).astype(np.uint16)

            gray = (ir >> 8).astype(np.uint8)
            gray_shape = (w, h)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, board, flags)
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                cv2.drawChessboardCorners(disp, board, corners2, True)
                if len(imgpoints) == 0 or (len(imgpoints) > 0 and time.time() % 0.5 < 0.03):
                    objpoints.append(objp.copy())
                    imgpoints.append(corners2)
                    print(f"Kinect v2 detections: {len(imgpoints)}/{min_samples}")

            cv2.imshow(win, disp)
            k = cv2.waitKey(WAIT_MS) & 0xFF
            if k == 27:  # ESC
                break
            if len(imgpoints) >= min_samples:
                break
    finally:
        cv2.destroyWindow(win)
        try:
            k2.close()
        except Exception:
            pass

    return objpoints, imgpoints, gray_shape


def _calibrate(objpoints, imgpoints, image_size):
    if not objpoints or not imgpoints or image_size is None:
        return None, None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Compute reprojection error
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_pts += n
    mean_rmse = np.sqrt(total_err / max(total_pts, 1))

    return ret, mtx, dist, rvecs, tvecs, float(mean_rmse)


def _save_intrinsics(path, camera_name, mtx, dist, image_size, board, square_size, reproj_error):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "camera_name": camera_name,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "checkerboard": {"cols": int(board[0]), "rows": int(board[1]), "square_size": float(square_size)},
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.reshape(-1).tolist(),
        "reprojection_error_rmse": reproj_error,
        "model": "pinhole+radial-tangential",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def _calibrate_from_images_or_live(name, img_glob, live_collector, json_path):
    # Try images first
    image_paths = sorted(glob.glob(img_glob))
    if image_paths:
        print(f"{name}: Found {len(image_paths)} images. Calibrating from files...")
        objp, imgp, size = _detect_corners_in_images(image_paths, CHECKERBOARD, SQUARE_SIZE)
        print("size:", size)
    else:
        print(f"{name}: No images found at {img_glob}. Falling back to live capture...")
        objp, imgp, size = live_collector(CHECKERBOARD, SQUARE_SIZE, MIN_SAMPLES)

    if not objp or not imgp or size is None:
        print(f"{name}: Not enough data to calibrate.")
        if not objp:
            print(f"{name}: No object points collected.")
        if not imgp:
            print(f"{name}: No image points collected.")
        return False

    ret, mtx, dist, rvecs, tvecs, rmse = _calibrate(objp, imgp, size)
    if not ret:
        print(f"{name}: Calibration failed.")
        return False

    print(f"{name} camera matrix:\n{mtx}")
    print(f"{name} dist coeffs:\n{dist.reshape(-1)}")
    print(f"{name} reprojection RMSE: {rmse:.4f}")

    _save_intrinsics(json_path, name, mtx, dist, size, CHECKERBOARD, SQUARE_SIZE, rmse)
    print(f"{name}: Saved intrinsics to {json_path}")
    return True


def main():
    ok_azure = _calibrate_from_images_or_live(
        "AzureKinectDK_IR",
        AZURE_IR_GLOB,
        _collect_live_azure,
        AZURE_JSON
    )

    ok_kv2 = _calibrate_from_images_or_live(
        "KinectV2_IR",
        KV2_IR_GLOB,
        _collect_live_kinect_v2,
        KV2_JSON
    )

    if not ok_azure and not ok_kv2:
        print("No calibration completed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())