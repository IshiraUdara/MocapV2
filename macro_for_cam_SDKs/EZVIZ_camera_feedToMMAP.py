import cv2
import mmap
import struct
import numpy as np
import time

# ======================
# CONFIG
# ======================
RTSP_URL = "rtsp://admin:WOJWUD@169.254.27.194:554/h264"

WIDTH = 1280
HEIGHT = 720
CHANNELS = 4  # RGBA

HEADER_SIZE = 16  # 4 ints
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
MMAP_SIZE = HEADER_SIZE + FRAME_SIZE

MMAP_NAME = "unity_rtsp_shared"

def main():
    # Create Windows named shared memory (no file on disk)
    mm = mmap.mmap(-1, MMAP_SIZE, tagname=MMAP_NAME, access=mmap.ACCESS_WRITE)

    # ======================
    # OPEN RTSP
    # ======================
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    frame_counter = 0
    print("RTSP → Shared Memory started")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                time.sleep(0.1)
                continue

            # Resize for fixed buffer
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # BGR → RGBA and flip vertically (upside down)
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_rgba = cv2.flip(frame_rgba, 0)

            # ======================
            # WRITE HEADER
            # ======================
            mm.seek(0)
            mm.write(struct.pack("iiii", WIDTH, HEIGHT, CHANNELS, frame_counter))

            # ======================
            # WRITE FRAME
            # ======================
            mm.write(frame_rgba.tobytes())

            frame_counter += 1
    finally:
        try: cap.release()
        except Exception: pass
        try: mm.close()
        except Exception: pass

if __name__ == "__main__":
    main()
