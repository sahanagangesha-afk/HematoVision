"""Image preprocessing shared by training-time checks and inference."""
import numpy as np
from PIL import Image

from .config import IMG_SIZE


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a (1, H, W, 3) float32 batch in the 0-255 range.

    Rescaling to 0-1 happens inside the model (Rescaling layer), so training
    and inference stay consistent.
    """
    img = img.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)
