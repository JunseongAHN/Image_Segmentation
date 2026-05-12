"""Create a small synthetic image for segmentation testing."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

Path("samples").mkdir(exist_ok=True)

h, w = 48, 48
image = np.zeros((h, w, 3), dtype=np.uint8)
image[:, :] = [40, 70, 110]
image[:, 24:] = [215, 190, 80]

# Add a small contrasting object.
y, x = np.ogrid[:h, :w]
circle = (x - 15) ** 2 + (y - 25) ** 2 < 9 ** 2
image[circle] = [220, 75, 90]

# Add light noise so the example is less artificial.
rng = np.random.default_rng(7)
noise = rng.normal(0, 6, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

imageio.imwrite("samples/sample_blocks.png", image)
print("Saved samples/sample_blocks.png")
