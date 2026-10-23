# RootSIFT extraction
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import HAND, GALLERY_DIR

# Create an OpenCV SIFT detector
def make_sift(nfeatures: int = HAND.nfeatures, contrast: float = HAND.contrast_threshold):
    return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)

def root_sift(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    desc /= desc.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(desc)

# Extracting Key Points and Descriptors from Grayscale Images
def extract_array(gray: np.ndarray, mask: np.ndarray | None = None,
                  nfeatures: int = HAND.nfeatures):
    sift = make_sift(nfeatures)
    kps, desc = sift.detectAndCompute(gray, mask)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, 128), np.float32)
    points = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kps], dtype=np.float32)
    return points, root_sift(desc)

# Read a grayscale image
def imread_gray(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# Read the image and extract RootSIFT features
def extract_image(path: str | Path, mask: np.ndarray | None = None,
                  nfeatures: int = HAND.nfeatures):
    gray = imread_gray(path)
    if gray is None:
        raise FileNotFoundError(path)
    return extract_array(gray, mask, nfeatures)

def extract_gallery_image(gid: int):
    path = GALLERY_DIR / f"{gid}.jpg"
    points, desc = extract_image(path)
    return int(gid), points, desc
