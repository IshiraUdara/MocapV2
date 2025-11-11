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

# Try to import a Spout receiver library. Prefer SpoutGL, then other common names.
_spout_backend = None
_spout_mod = None
try:
    import SpoutGL as spout  # SpoutGL C-extension wrapper
    _spout_mod = spout
    _spout_backend = 'SpoutGL'
except Exception:
    try:
        import spout  # fallback name if installed under 'spout'
        _spout_mod = spout
        _spout_backend = 'spout'
    except Exception:
        try:
            from Spout import SpoutReceiver as _SpoutReceiver  # other wrapper
            _spout_mod = None
            _spout_backend = 'Spout'
        except Exception:
            _spout_backend = None
            _spout_mod = None

class SpoutReceiver:
    """
    SpoutGL-aware receiver: tries the SpoutGL object's methods in a few common ways
    to obtain an image buffer and return a numpy BGR image (H,W,3) or None.
    """
    def __init__(self, sender_name):
        self.sender_name = sender_name
        self.width = 0
        self.height = 0
        self._backend = _spout_backend
        self._client = None
        self._client_info = None

        if self._backend in ('SpoutGL', 'spout') and _spout_mod is not None:
            mod = _spout_mod
            try:
                attrs = [a for a in dir(mod) if not a.startswith('_')]
                print(f"spout module attrs: {attrs}")
            except Exception:
                pass

            candidate = getattr(mod, 'SpoutReceiver', None)
            if callable(candidate):
                try:
                    try:
                        client = candidate(self.sender_name)
                    except TypeError:
                        client = candidate()
                    self._client = client
                    self._client_info = f"{_spout_backend}.SpoutReceiver"
                    print(f"Initialized {_spout_backend}.SpoutReceiver() for '{self.sender_name}'")
                except Exception as e:
                    print(f"{_spout_backend}.SpoutReceiver construction failed: {e}")
        elif self._backend == 'Spout':
            try:
                self._client = _SpoutReceiver(self.sender_name)
                self._client_info = "Spout.SpoutReceiver"
                print(f"Initialized Spout.SpoutReceiver for '{self.sender_name}'")
            except Exception as e:
                print(f"Spout backend init failed: {e}")
                self._client = None
        else:
            print("No Spout Python binding found. Placeholders will be shown.")

    def _bytes_to_bgr(self, buf, w, h, fmt_hint=None):
        """Convert raw bytes or memoryview to BGR numpy array, guessing channels from fmt_hint."""
        try:
            if isinstance(buf, memoryview):
                buf = buf.tobytes()
            arr = np.frombuffer(buf, dtype=np.uint8)
            # guess channels: if length matches w*h*3 or w*h*4
            if arr.size == w * h * 3:
                arr = arr.reshape((h, w, 3))
                # assume RGB -> convert to BGR
                try:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except Exception:
                    return arr
            if arr.size == w * h * 4:
                arr = arr.reshape((h, w, 4))
                # drop alpha and convert RGB->BGR if needed
                rgb = arr[:, :, :3]
                try:
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception:
                    return rgb
        except Exception:
            pass
        return None

    def receive(self):
        """Return BGR numpy array or None. Uses SpoutGL receiver's concrete API parsing."""
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

        # if not connected, nothing to do
        try:
            if hasattr(c, 'isConnected') and callable(c.isConnected):
                try:
                    if not c.isConnected():
                        return None
                except Exception:
                    pass
        except Exception:
            pass

        w, h = int(self.width or 0), int(self.height or 0)

        # If API reports frames availability, prefer that check
        try:
            if hasattr(c, 'isFrameNew') and callable(c.isFrameNew):
                try:
                    # only proceed if a new frame exists
                    if not c.isFrameNew():
                        return None
                except Exception:
                    pass
        except Exception:
            pass

        # 1) Try receiveImage() which commonly returns (ok, buf, w, h) or (buf,w,h) or ndarray
        if hasattr(c, 'receiveImage'):
            try:
                out = None
                try:
                    out = c.receiveImage()
                except TypeError:
                    try:
                        out = c.receiveImage(self.sender_name)
                    except Exception:
                        out = None

                if out is not None:
                    # numpy array -> assume RGB, convert to BGR
                    if isinstance(out, np.ndarray):
                        if out.ndim == 3 and out.shape[2] == 3:
                            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        return out

                    # tuple/list patterns
                    if isinstance(out, (tuple, list)):
                        # pattern: (ok, buf, w, h)
                        if len(out) >= 4:
                            ok, buf, maybe_w, maybe_h = out[0], out[1], int(out[2]), int(out[3])
                            if isinstance(ok, bool) and not ok:
                                return None
                            if isinstance(buf, (bytes, bytearray, memoryview)):
                                img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                                if img is not None:
                                    return img
                            if isinstance(buf, np.ndarray):
                                return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR) if buf.ndim == 3 and buf.shape[2] == 3 else buf
                        # pattern: (buf, w, h)
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
                pass

        # 2) Try readMemoryBuffer() which often returns raw bytes or (buf,w,h)
        if hasattr(c, 'readMemoryBuffer'):
            try:
                out = None
                try:
                    out = c.readMemoryBuffer()
                except TypeError:
                    try:
                        out = c.readMemoryBuffer(self.sender_name)
                    except Exception:
                        out = None

                if out is not None:
                    if isinstance(out, (bytes, bytearray, memoryview)):
                        if w and h:
                            img = self._bytes_to_bgr(out, w, h)
                            if img is not None:
                                return img
                    if isinstance(out, (tuple, list)) and len(out) >= 3:
                        buf = out[0]; maybe_w = int(out[1]); maybe_h = int(out[2])
                        img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                        if img is not None:
                            return img
                    if isinstance(out, np.ndarray):
                        if out.ndim == 3 and out.shape[2] == 3:
                            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        return out
            except Exception:
                pass

        # 3) Try receiveTexture() -> may return bytes/tuple similarly
        if hasattr(c, 'receiveTexture'):
            try:
                out = None
                try:
                    out = c.receiveTexture()
                except TypeError:
                    try:
                        out = c.receiveTexture(self.sender_name)
                    except Exception:
                        out = None

                if out is not None:
                    if isinstance(out, (tuple, list)) and len(out) >= 3:
                        buf = out[0]; maybe_w = int(out[-2]); maybe_h = int(out[-1])
                        img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                        if img is not None:
                            return img
                    if isinstance(out, np.ndarray):
                        if out.ndim == 3 and out.shape[2] == 3:
                            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        return out
            except Exception:
                pass

        # 4) Fall back: use getSenderInfo/getSenderFrame to extract buffer if available
        try:
            if hasattr(c, 'getSenderFrame'):
                try:
                    frame = c.getSenderFrame()
                    # frame may be bytes/ndarray or tuple (buf,w,h)
                    if frame is not None:
                        if isinstance(frame, np.ndarray):
                            if frame.ndim == 3 and frame.shape[2] == 3:
                                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            return frame
                        if isinstance(frame, (bytes, bytearray, memoryview)):
                            if w and h:
                                img = self._bytes_to_bgr(frame, w, h)
                                if img is not None:
                                    return img
                        if isinstance(frame, (tuple, list)) and len(frame) >= 3:
                            buf = frame[0]; maybe_w = int(frame[1]); maybe_h = int(frame[2])
                            img = self._bytes_to_bgr(buf, maybe_w, maybe_h)
                            if img is not None:
                                return img
                except Exception:
                    pass
        except Exception:
            pass

        # no image obtained
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
                                                            lines=["No Spout binding found.",
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

def run_two_spout_senders(sender_names, flipped=True, floor=False, auto=False):
    global running, take_photo

    if not isinstance(sender_names, (list, tuple)) or len(sender_names) < 1:
        print("Please provide at least one sender name.")
        return False

    shared_frames = {}
    lock = threading.Lock()
    threads = []

    for name in sender_names:
        shared_frames[name] = (make_placeholder(name), time.time())

    for name in sender_names:
        t = ReceiverThread(name, shared_frames, lock)
        t.start()
        threads.append(t)

    for idx, name in enumerate(sender_names):
        if floor:
            os.makedirs(f'Kinect_DK_cam/tracking/cam{idx}', exist_ok=True)
        else:
            os.makedirs(f'Kinect_DK_cam/captured_images/cam{idx}', exist_ok=True)

    window_names = [f'Spout Feed - {n}' for n in sender_names]
    for wn in window_names:
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)

    fps_counters = {n: {'count':0, 'start':time.time(), 'fps':0.0} for n in sender_names}

    try:
        while running.is_set():
            display_any = False
            with lock:
                items = list(shared_frames.items())
            for idx, (sender, data) in enumerate(items):
                frame_bgr, ts = data
                if frame_bgr is None:
                    frame_bgr = make_placeholder(sender)
                if flipped:
                    disp = cv2.flip(frame_bgr, 1)
                else:
                    disp = frame_bgr

                c = fps_counters.get(sender)
                if c is not None:
                    c['count'] += 1
                    if c['count'] >= 30:
                        now = time.time()
                        c['fps'] = c['count'] / (now - c['start']) if (now - c['start']) > 0 else 0.0
                        c['count'] = 0
                        c['start'] = now
                    fps_text = f"FPS: {c['fps']:.1f}"
                else:
                    fps_text = "FPS: 0.0"

                cv2.putText(disp, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                win_name = window_names[sender_names.index(sender)]
                cv2.imshow(win_name, disp)
                display_any = True

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                running.clear()
                break
            elif key == ord('s'):
                with lock:
                    for idx, (sender, data) in enumerate(shared_frames.items()):
                        frame_bgr, ts = data
                        timestamp = int(ts)
                        if floor:
                            fname = f'Kinect_DK_cam/tracking/cam{idx}/{timestamp}.png'
                        else:
                            fname = f'Kinect_DK_cam/captured_images/cam{idx}/{timestamp}.png'
                        cv2.imwrite(fname, frame_bgr)
                        print(f"Saved {fname}")
            if not display_any:
                time.sleep(0.02)

            if auto:
                take_photo.set()
                time.sleep(0.1)
                take_photo.clear()

    except KeyboardInterrupt:
        running.clear()
    finally:
        for wn in window_names:
            try:
                cv2.destroyWindow(wn)
            except Exception:
                pass

    for t in threads:
        t.join(timeout=1.0)

    print("All Spout receiver threads stopped")
    return True

def main():
    TWO_SENDER_NAMES = ["kinectazure_output", "kinect_xbox_output"]
    flipped = True
    floor = False
    auto = False

    try:
        print("Starting TouchDesigner Spout receivers for Kinect feeds")
        print("Ensure in TouchDesigner Spout Out TOP the Sender Name matches one of:")
        for s in TWO_SENDER_NAMES:
            print(f"  - {s}")
        success = run_two_spout_senders(TWO_SENDER_NAMES, flipped=flipped, floor=floor, auto=auto)
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