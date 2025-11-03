import sys
import time
import threading
import cv2
import os
import numpy as np

running = threading.Event()
running.set()
take_photo = threading.Event()
take_photo.clear()

# Use only SpoutGL for Spout integration
_spout_backend = None
_spout_mod = None
try:
    import SpoutGL as spout  # required SpoutGL C-extension wrapper
    _spout_mod = spout
    _spout_backend = 'SpoutGL'
except Exception as e:
    print(f"SpoutGL import failed: {e}")
    _spout_mod = None
    _spout_backend = None

class SpoutReceiver:
    """
    SpoutGL-aware receiver: uses SpoutGL API to obtain an image buffer
    and return a numpy BGR image (H,W,3) or None.
    This variant forces receiver name activation and emits basic diagnostics.
    """
    def __init__(self, sender_name):
        self.sender_name = sender_name
        self.width = 0
        self.height = 0
        self._backend = _spout_backend
        self._client = None
        self._client_info = None

        if self._backend != 'SpoutGL' or _spout_mod is None:
            print("SpoutGL not available; will show placeholders.")
            return

        mod = _spout_mod
        candidate = getattr(mod, 'SpoutReceiver', None)
        if not callable(candidate):
            print("SpoutGL.SpoutReceiver not found in SpoutGL module.")
            return

        try:
            # Prefer constructor with sender name where supported
            try:
                client = candidate(self.sender_name)
            except TypeError:
                client = candidate()
            self._client = client
            self._client_info = "SpoutGL.SpoutReceiver"
            print(f"Initialized SpoutGL.SpoutReceiver() for '{self.sender_name}'")
        except Exception as e:
            print(f"SpoutGL.SpoutReceiver construction failed: {e}")
            self._client = None
            return

        # Try to explicitly set receiver / active sender and initialize GL where supported
        try:
            if hasattr(self._client, 'setReceiverName'):
                try:
                    self._client.setReceiverName(self.sender_name)
                    print("Called setReceiverName")
                except Exception as e:
                    print("setReceiverName failed:", e)
            if hasattr(self._client, 'setActiveSender'):
                try:
                    # setActiveSender may accept name or index
                    self._client.setActiveSender(self.sender_name)
                    print("Called setActiveSender")
                except Exception as e:
                    # ignore if not supported with name
                    print("setActiveSender failed:", e)
            if hasattr(self._client, 'createOpenGL'):
                try:
                    self._client.createOpenGL()
                    print("Called createOpenGL()")
                except Exception as e:
                    print("createOpenGL failed:", e)
            # print available senders
            if hasattr(self._client, 'getSenderList'):
                try:
                    s = self._client.getSenderList()
                    print("Spout sender list:", s)
                except Exception as e:
                    print("getSenderList() failed:", e)
            # print active sender name if available
            if hasattr(self._client, 'getSenderName'):
                try:
                    name = self._client.getSenderName()
                    print("Receiver initial sender name:", name)
                except Exception as e:
                    pass
        except Exception:
            pass

    def _bytes_to_bgr(self, buf, w, h, fmt_hint=None):
        try:
            if isinstance(buf, memoryview):
                buf = buf.tobytes()
            arr = np.frombuffer(buf, dtype=np.uint8)
            if arr.size == w * h * 3:
                arr = arr.reshape((h, w, 3))
                try:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except Exception:
                    return arr
            if arr.size == w * h * 4:
                arr = arr.reshape((h, w, 4))
                rgb = arr[:, :, :3]
                try:
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception:
                    return rgb
        except Exception:
            pass
        return None

    def receive(self):
        """Return BGR numpy array or None. Emits brief diagnostics when no frame found."""
        if self._client is None:
            return None

        c = self._client

        # update known sender size if available
        try:
            if hasattr(c, 'getSenderWidth'):
                w = int(c.getSenderWidth() or 0)
            else:
                w = int(getattr(c, 'width', 0) or self.width or 0)
            if hasattr(c, 'getSenderHeight'):
                h = int(c.getSenderHeight() or 0)
            else:
                h = int(getattr(c, 'height', 0) or self.height or 0)
            if w > 0 and h > 0:
                self.width, self.height = w, h
        except Exception:
            w, h = int(self.width or 0), int(self.height or 0)

        # Quick diagnostics for connection / frame state
        try:
            if hasattr(c, 'isConnected'):
                try:
                    conn = c.isConnected()
                    if not conn:
                        # no connected sender
                        # print once per null connection to avoid spam
                        # but we still return None so placeholder shows
                        # user can check TouchDesigner Sender Name and Spout Out TOP
                        # print("SpoutGL: isConnected() ->", conn)
                        return None
                except Exception:
                    pass
            if hasattr(c, 'isFrameNew'):
                try:
                    frame_new = c.isFrameNew()
                    if not frame_new:
                        return None
                except Exception:
                    pass
        except Exception:
            pass

        w, h = int(self.width or 0), int(self.height or 0)

        # Try the typical CPU APIs in order
        # receiveImage -> readMemoryBuffer -> receiveTexture -> getSenderFrame
        for fn_name in ('receiveImage', 'readMemoryBuffer', 'receiveTexture', 'getSenderFrame'):
            fn = getattr(c, fn_name, None)
            if not callable(fn):
                continue
            try:
                out = None
                try:
                    out = fn()
                except TypeError:
                    try:
                        out = fn(self.sender_name)
                    except Exception:
                        out = None
                if out is None:
                    continue
                # ndarray returned
                if isinstance(out, np.ndarray):
                    if out.ndim == 3 and out.shape[2] == 3:
                        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    return out
                # tuple (ok, buf, w, h) or (buf,w,h)
                if isinstance(out, (tuple, list)):
                    if len(out) >= 4:
                        ok, buf, maybe_w, maybe_h = out[0], out[1], int(out[2]), int(out[3])
                        if isinstance(ok, bool) and not ok:
                            continue
                        if isinstance(buf, (bytes, bytearray, memoryview)):
                            img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                            if img is not None:
                                return img
                        if isinstance(buf, np.ndarray):
                            return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR) if buf.ndim == 3 and buf.shape[2] == 3 else buf
                    if len(out) == 3:
                        buf, maybe_w, maybe_h = out[0], int(out[1]), int(out[2])
                        if isinstance(buf, (bytes, bytearray, memoryview)):
                            img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                            if img is not None:
                                return img
                        if isinstance(buf, np.ndarray):
                            return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR) if buf.ndim == 3 and buf.shape[2] == 3 else buf
                # raw bytes
                if isinstance(out, (bytes, bytearray, memoryview)):
                    if w and h:
                        img = self._bytes_to_bgr(out, w, h)
                        if img is not None:
                            return img
            except Exception:
                # try next API
                continue

        # nothing returned
        return None

def make_placeholder(sender_name, w=640, h=480, lines=None):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    title = f"Waiting for: {sender_name}"
    cv2.putText(img, title, (10, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if lines:
        y = h//2 + 10
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            y += 25
    return img

# Receiver thread: captures frames and timestamps, does NOT call cv2.imshow or waitKey
class ReceiverThread(threading.Thread):
    def __init__(self, sender_name, shared_dict, lock, poll_interval=0.005):
        super().__init__(daemon=True)
        self.sender = sender_name
        self.shared = shared_dict
        self.lock = lock
        self.poll_interval = poll_interval
        try:
            self.receiver = SpoutReceiver(sender_name)
        except Exception as e:
            print(f"Failed to init SpoutReceiver for '{sender_name}': {e}")
            self.receiver = None

    def run(self):
        if self.receiver is None:
            with self.lock:
                self.shared[self.sender] = (make_placeholder(self.sender, 640, 480,
                                                            lines=["No SpoutGL binding found.",
                                                                   "Set Spout Out TOP Sender Name."]), time.time())
            while running.is_set():
                time.sleep(0.5)
            return

        while running.is_set():
            try:
                frame = self.receiver.receive()
                if frame is not None:
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        try:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        except Exception:
                            frame_bgr = frame
                    elif frame.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame

                    ts = time.time()
                    with self.lock:
                        self.shared[self.sender] = (frame_bgr, ts)
                else:
                    with self.lock:
                        if self.sender not in self.shared:
                            self.shared[self.sender] = (make_placeholder(self.sender), time.time())
                    time.sleep(self.poll_interval)
            except Exception as ex:
                print(f"Receiver '{self.sender}' error: {ex}")
                time.sleep(0.05)

def run_single_spout_sender(sender_name, flipped=True, floor=False, auto=False):
    global running, take_photo

    shared_frames = {}
    lock = threading.Lock()

    # placeholder so window appears immediately
    shared_frames[sender_name] = (make_placeholder(sender_name), time.time())

    t = ReceiverThread(sender_name, shared_frames, lock)
    t.start()

    # prepare folders (cam0)
    cam_idx = 0
    if floor:
        os.makedirs(f'Kinect_DK_cam/tracking/cam{cam_idx}', exist_ok=True)
    else:
        os.makedirs(f'Kinect_DK_cam/captured_images/cam{cam_idx}', exist_ok=True)

    window_name = f'Spout Feed - {sender_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_counter = {'count': 0, 'start': time.time(), 'fps': 0.0}

    try:
        while running.is_set():
            with lock:
                frame_bgr, ts = shared_frames.get(sender_name, (make_placeholder(sender_name), time.time()))

            if frame_bgr is None:
                frame_bgr = make_placeholder(sender_name)

            if flipped:
                disp = cv2.flip(frame_bgr, 1)
            else:
                disp = frame_bgr

            # simple FPS
            fps_counter['count'] += 1
            if fps_counter['count'] >= 30:
                now = time.time()
                fps_counter['fps'] = fps_counter['count'] / (now - fps_counter['start']) if (now - fps_counter['start']) > 0 else 0.0
                fps_counter['count'] = 0
                fps_counter['start'] = now

            cv2.putText(disp, f"FPS: {fps_counter['fps']:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow(window_name, disp)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                running.clear()
                break
            elif key == ord('s'):
                # save current frame
                timestamp = int(ts)
                if floor:
                    fname = f'Kinect_DK_cam/tracking/cam{cam_idx}/{timestamp}.png'
                else:
                    fname = f'Kinect_DK_cam/captured_images/cam{cam_idx}/{timestamp}.png'
                cv2.imwrite(fname, frame_bgr)
                print(f"Saved {fname}")

            if auto:
                take_photo.set()
                time.sleep(0.1)
                take_photo.clear()

            # small sleep to reduce CPU when no new frames
            time.sleep(0.001)

    except KeyboardInterrupt:
        running.clear()
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    t.join(timeout=1.0)
    print("Spout receiver thread stopped")
    return True

def main():
    # change this to the Sender Name you set in TouchDesigner Spout Out TOP
    SINGLE_SENDER_NAME = "kinectazure_output"

    flipped = True
    floor = False
    auto = False

    try:
        print("Starting TouchDesigner Spout receiver for Azure Kinect DK feed")
        print(f"Ensure TouchDesigner Spout Out TOP Sender Name is: {SINGLE_SENDER_NAME}")
        success = run_single_spout_sender(SINGLE_SENDER_NAME, flipped=flipped, floor=floor, auto=auto)
        return success
    except Exception as e:
        print(f"Error in main: {e}")
        return False

if __name__ == '__main__':
    try:
        ok = main()
        print("Exiting...")
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        running.clear()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)