# scripts/graph_delta/mask_ops.py
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None


# ---------------- target category heuristics ----------------
SLENDER_WORDS = {
    "spoon", "fork", "knife", "chopsticks", "tongs", "ladle", "spatula",
    "whisk", "peeler", "straw", "brush"
}
CONTAINER_WORDS = {
    "bowl", "cup", "mug", "glass", "plate", "saucer", "dish", "tray",
    "ramekin", "pot", "pan", "basket", "vase", "jar", "bottle"
}


def guess_target_category(t: str) -> str:
    t = (t or "").lower().strip()
    if t in SLENDER_WORDS:
        return "slender"
    if t in CONTAINER_WORDS:
        return "container"
    return "generic"


def add_mask_params_for_target(category: str) -> Dict:
    if category == "slender":
        return dict(pad_w_ratio=0.35, pad_h_ratio=0.10, max_pad_ratio=0.55, min_pad=10)
    if category == "container":
        return dict(pad_w_ratio=0.28, pad_h_ratio=0.18, max_pad_ratio=0.60, min_pad=10)
    return dict(pad_w_ratio=0.30, pad_h_ratio=0.14, max_pad_ratio=0.58, min_pad=10)


# ---------------- bbox helpers ----------------
def _clamp_bbox_xyxy(b: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(W - 1, int(x0)))
    y0 = max(0, min(H - 1, int(y0)))
    x1 = max(1, min(W, int(x1)))
    y1 = max(1, min(H, int(y1)))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def bbox_from_mask_np(mask_np: np.ndarray, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return (0, 0, W, H)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0 -= int(pad); y0 -= int(pad)
    x1 += int(pad); y1 += int(pad)
    return _clamp_bbox_xyxy((x0, y0, x1, y1), W, H)


def align_bbox_to_multiple(b: Tuple[int, int, int, int], W: int, H: int, mult: int = 8) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b

    def down(v): return int(v // mult * mult)
    def up(v): return int((v + mult - 1) // mult * mult)

    x0 = down(x0); y0 = down(y0)
    x1 = up(x1);   y1 = up(y1)
    return _clamp_bbox_xyxy((x0, y0, x1, y1), W, H)


def get_bbox_from_mask(mask_np: np.ndarray, expand: int = 2) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError("mask is empty, cannot compute bbox.")
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0 = max(0, x0 - expand)
    y0 = max(0, y0 - expand)
    x1 = x1 + expand
    y1 = y1 + expand
    return x0, y0, x1, y1


# ---------------- morphology ----------------
def refine_mask_binary(mask_np: np.ndarray, dilate: int = 2, erode: int = 0) -> np.ndarray:
    if cv2 is None:
        return (mask_np > 0).astype(np.uint8)
    m = (mask_np > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if dilate > 0:
        m = cv2.dilate(m, kernel, iterations=int(dilate))
    if erode > 0:
        m = cv2.erode(m, kernel, iterations=int(erode))
    return (m > 0).astype(np.uint8)


def make_inpaint_mask_pil(mask_np: np.ndarray, dilate: int = 6, blur: int = 11, erode: int = 0) -> Image.Image:
    m_bin = refine_mask_binary(mask_np, dilate=dilate, erode=erode)
    m = (m_bin > 0).astype(np.uint8) * 255
    if cv2 is None:
        return Image.fromarray(m, mode="L")
    if blur and blur > 0:
        k = int(blur)
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return Image.fromarray(m, mode="L")


def feather_mask(mask_np01: np.ndarray, feather: int = 5) -> np.ndarray:
    if feather <= 0 or cv2 is None:
        return mask_np01.astype(np.float32)
    m = (mask_np01 > 0).astype(np.uint8) * 255
    k = feather * 2 + 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    return (m.astype(np.float32) / 255.0)


def make_ring_mask(mask_np: np.ndarray, dilate: int = 10, erode: int = 3, blur: int = 9) -> Image.Image:
    if cv2 is None:
        m = (mask_np > 0).astype(np.uint8) * 255
        return Image.fromarray(m, mode="L")
    m = (mask_np > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    md = cv2.dilate(m, k, iterations=int(dilate))
    me = cv2.erode(m, k, iterations=int(erode))
    ring = cv2.subtract(md, me)
    if blur and blur > 0:
        kk = blur if blur % 2 == 1 else blur + 1
        ring = cv2.GaussianBlur(ring, (kk, kk), 0)
    return Image.fromarray(ring, mode="L")


# ---------------- replace/add mask shaping ----------------
def make_add_mask_for_replace_bbox_clamped(
    mask_np: np.ndarray,
    W: int,
    H: int,
    pad_w_ratio: float = 0.45,
    pad_h_ratio: float = 0.12,
    min_pad: int = 12,
    max_pad_ratio: float = 0.65,
    blur: int = 0,
) -> Image.Image:
    x0, y0, x1, y1 = get_bbox_from_mask(mask_np, expand=0)
    bw, bh = (x1 - x0), (y1 - y0)
    pad_x = int(bw * pad_w_ratio)
    pad_y = int(bh * pad_h_ratio)
    pad_x = max(min_pad, min(pad_x, int(max_pad_ratio * bw)))
    pad_y = max(min_pad // 2, min(pad_y, int(max_pad_ratio * bh)))
    xx0 = max(0, x0 - pad_x)
    xx1 = min(W, x1 + pad_x)
    yy0 = max(0, y0 - pad_y)
    yy1 = min(H, y1 + pad_y)

    m = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(m)
    draw.rectangle([xx0, yy0, xx1, yy1], fill=255)

    if cv2 is not None and blur and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        mm = cv2.GaussianBlur(np.array(m), (k, k), 0)
        m = Image.fromarray(mm.astype(np.uint8), mode="L")
    return m


def fill_mask_region_with_bg_noise(image_pil, mask_pil, seed=0, noise_std=8, blur=9):
    img = np.array(image_pil.convert("RGB")).astype(np.float32)
    m = (np.array(mask_pil.convert("L")) > 128)

    bg = img[~m]
    if bg.size == 0:
        bg_mean = img.mean(axis=(0, 1))
    else:
        bg_mean = bg.mean(axis=0)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    noise = rng.normal(0, noise_std, size=img.shape).astype(np.float32)

    out = img.copy()
    out[m] = bg_mean + noise[m]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out)

    if cv2 is not None and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        out_np = cv2.GaussianBlur(np.array(out_pil), (k, k), 0)
        out_pil = Image.fromarray(out_np)
    return out_pil
