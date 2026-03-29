# graph/mask_utils.py
from typing import Tuple, List
import numpy as np

def bbox_xywh_to_xyxy(x: int, y: int, w: int, h: int) -> List[int]:
    return [int(x), int(y), int(x + w), int(y + h)]

def clamp_bbox_xyxy(b: List[int], W: int, H: int) -> List[int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return [x0, y0, x1, y1]

def bbox_to_mask_xyxy(bbox_xyxy: List[int], W: int, H: int) -> np.ndarray:
    """Return float32 mask [H,W] with 1 inside bbox."""
    x0, y0, x1, y1 = bbox_xyxy
    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m
