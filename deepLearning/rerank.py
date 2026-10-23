# Regional matching on cached feature maps via integral images.
from __future__ import annotations

import numpy as np
import torch

from config import DEEP

# Calculates the prefix sum of x^p channel-by-channel for feature maps,
def integral_map(fmap: torch.Tensor, p: float = DEEP.gem_p) -> torch.Tensor:
    x = fmap.clamp(min=1e-6).pow(p)
    cumulative = x.cumsum(-1).cumsum(-2)
    out = torch.zeros(
        (cumulative.shape[0], cumulative.shape[1] + 1, cumulative.shape[2] + 1),
        dtype=cumulative.dtype,
    )
    out[:, 1:, 1:] = cumulative
    return out

# Generates the starting point of a window in a given direction
def _positions(total: int, size: int, stride: float) -> list[int]:
    step = max(1, int(round(size * stride)))
    last = total - size
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions

# Enumerates candidate windows
def window_boxes(
    hf: int,
    wf: int,
    base_aspect: float = 1.0,
    scales=DEEP.window_scales,
    aspect_mults=DEEP.window_aspect_mults,
    stride: float = DEEP.window_stride,
    max_windows: int = DEEP.window_max,
    min_cells: int = DEEP.window_min_cells,
) -> list[tuple[int, int, int, int]]:

    boxes: list[tuple[int, int, int, int]] = []
    for scale in scales:
        height = min(hf, max(min_cells, int(round(scale * hf))))
        for mult in aspect_mults:
            width = min(wf, max(min_cells, int(round(height * base_aspect * mult))))
            for y in _positions(hf, height, stride):
                for x in _positions(wf, width, stride):
                    boxes.append((x, y, x + width, y + height))
    if len(boxes) > max_windows:
        keep = np.unique(np.linspace(0, len(boxes) - 1, max_windows).round().astype(int))
        boxes = [boxes[i] for i in keep]
    return boxes


# Uses an integration map to pool the descriptors (N, C) from all windows in a single step.
def pool_windows(integral: torch.Tensor, boxes, p: float = DEEP.gem_p) -> torch.Tensor:
    b = torch.as_tensor(np.asarray(boxes), dtype=torch.long)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    sums = (
        integral[:, y1, x1]
        - integral[:, y0, x1]
        - integral[:, y1, x0]
        + integral[:, y0, x0]
    )
    area = ((x1 - x0) * (y1 - y0)).to(sums.dtype).unsqueeze(0)
    return sums.clamp(min=0).pow(1.0 / p).div(area.pow(1.0 / p)).t()


# Calculates the intersection and union ratios of two rectangles
def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


# Calculates a softmax-weighted average of the top-scoring overlapping windows to suppress the single-window jitter caused by arg-max.
def refine_box(boxes, sims, top: int = 8, iou_threshold: float = 0.4, temperature: float = 0.03):
    order = np.argsort(-np.asarray(sims))[:top]
    best = boxes[int(order[0])]
    chosen = [int(i) for i in order if iou(best, boxes[int(i)]) >= iou_threshold]
    if len(chosen) <= 1:
        return tuple(int(round(v)) for v in best)

    values = np.asarray(sims)
    weights = np.exp((values[chosen] - np.max(values[chosen])) / temperature)
    weights /= weights.sum()
    arr = np.asarray([boxes[i] for i in chosen], dtype=np.float64)
    blended = (weights[:, None] * arr).sum(0)
    x0, y0, x1, y1 = (int(round(v)) for v in blended)
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))
