import os
import mmap
import numpy as np
from PIL import Image

# List your cameras (must match Unity mmfName and RT size)
cameras = [
    {"name": "UnityCam1", "width": 256, "height": 256, "channels": 4},
    {"name": "UnityCam2", "width": 256, "height": 256, "channels": 4},
]

OUT_DIR = "Unity_Simulation/texture_import_from_unity"
os.makedirs(OUT_DIR, exist_ok=True)

def read_once(cam):
    size = cam["width"] * cam["height"] * cam["channels"]
    try:
        # Windows named shared memory: tagname must match Unity's mmfName
        with mmap.mmap(-1, size, tagname=cam["name"], access=mmap.ACCESS_READ) as mmf:
            mmf.seek(0)
            data = mmf.read(size)
    except (FileNotFoundError, OSError) as e:
        print(f"[{cam['name']}] MMF not found/openable: {e}")
        return None, None

    if len(data) != size:
        print(f"[{cam['name']}] Unexpected size: got {len(data)} bytes, expected {size}")
        return None, None

    buf = np.frombuffer(data, dtype=np.uint8)
    nonzero = int(np.count_nonzero(buf))
    print(f"[{cam['name']}] bytes: {len(buf)}, nonzero: {nonzero}")

    if nonzero == 0:
        # Unity hasn’t written yet this frame; with the OnPostRender script this should go away.
        print(f"[{cam['name']}] Buffer is all zeros (black).")
        # Still return the black image so caller can save it if needed
    return data, nonzero

def to_images(cam, data):
    h, w, c = cam["height"], cam["width"], cam["channels"]
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))

    # Unity → bottom-left origin; flip vertically for PIL
    arr_flipped = np.flip(arr, axis=0)

    # Try RGBA as-is
    img_rgba = Image.fromarray(arr_flipped, 'RGBA')

    # Also try BGRA→RGBA swap in case the bytes are BGRA
    arr_bgra = arr_flipped[..., [2, 1, 0, 3]]
    img_from_bgra = Image.fromarray(arr_bgra, 'RGBA')

    return img_rgba, img_from_bgra

if __name__ == "__main__":
    for cam in cameras:
        data, nonzero = read_once(cam)
        if data is None:
            continue

        img_rgba, img_bgra = to_images(cam, data)

        # Save both so you can compare quickly
        p_rgba = os.path.join(OUT_DIR, f"{cam['name']}_rgba.png")
        p_bgra = os.path.join(OUT_DIR, f"{cam['name']}_bgra.png")
        img_rgba.save(p_rgba)
        img_bgra.save(p_bgra)

        # Heuristic: if RGBA version looks almost black but BGRA has variance, prefer BGRA
        if nonzero == 0:
            print(f"[{cam['name']}] Saved black buffers (likely no Unity frame yet): {p_rgba}, {p_bgra}")
        else:
            print(f"[{cam['name']}] Saved: {p_rgba} and {p_bgra}")
            # Optionally show one:
            # img_bgra.show(title=f"{cam['name']} (BGRA try)")
