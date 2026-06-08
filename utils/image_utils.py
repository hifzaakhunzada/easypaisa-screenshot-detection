"""
utils/image_utils.py
Common image loading, resizing, and preprocessing helpers used across modules.
"""

import cv2
import numpy as np
from PIL import Image


def load_image_cv2(image_path: str) -> np.ndarray:
    """Load an image as a BGR numpy array using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    return img


def load_image_pil(image_path: str) -> Image.Image:
    """Load an image as a PIL Image object."""
    return Image.open(image_path)


def preprocess_for_model(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize and normalise an image for CNN input.
    Returns a float32 array with shape (1, H, W, 3) ready for model.predict().
    """
    img = load_image_pil(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
    return np.expand_dims(arr, axis=0)               # add batch dimension


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, width: int = 800) -> np.ndarray:
    """Resize an OpenCV image to a given width while preserving aspect ratio."""
    h, w = image.shape[:2]
    ratio = width / w
    new_dims = (width, int(h * ratio))
    return cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)
