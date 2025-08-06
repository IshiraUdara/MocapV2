import mmap
import numpy as np
from PIL import Image

width = 256
height = 256
channels = 4
size = width * height * channels
mmf_name = "UnityToPythonMMF"

with mmap.mmap(-1, size, tagname=mmf_name, access=mmap.ACCESS_READ) as mmf:
    mmf.seek(0)
    data = mmf.read(size)

# Sanity check
print(f"Read {len(data)} bytes")

image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))
image = np.flip(image, axis=0)
Image.fromarray(image, 'RGBA').show()
Image.fromarray(image, 'RGBA').save("texture_import_from_unity/output_image.png")
