# Geometric verification
from __future__ import annotations

import cv2
import numpy as np

from config import HAND


# Lowe's ratio test
def ratio_matches(query_desc: np.ndarray, gallery_desc: np.ndarray,
                  ratio: float = HAND.ratio):
    if len(query_desc) < 4 or len(gallery_desc) < 4:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query_desc, gallery_desc, k=2)
    return [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]

# RANSAC Verification
def verify(query_points: np.ndarray, query_desc: np.ndarray,
           gallery_points: np.ndarray, gallery_desc: np.ndarray,
           query_box: tuple[int, int, int, int],
           ratio: float = HAND.ratio, thresh: float = HAND.ransac_thresh,
           min_inliers: int = HAND.min_inliers):
    good = ratio_matches(query_desc, gallery_desc, ratio)
    if len(good) < min_inliers:
        return 0, None

    src = np.float32([query_points[m.queryIdx, :2] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([gallery_points[m.trainIdx, :2] for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    if homography is None or mask is None:
        return 0, None

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return inliers, None

    x, y, w, h = query_box
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except cv2.error:
        return inliers, None
    if not np.all(np.isfinite(projected)):
        return inliers, None

    x0, y0 = projected.min(0)
    x1, y1 = projected.max(0)
    box = (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))
    if box[2] <= 0 or box[3] <= 0:
        return inliers, None
    return inliers, box
