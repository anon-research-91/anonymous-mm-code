# scripts/graph_delta/pipeline_replace.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image

from .pipeline_remove import run_remove
from .pipeline_add import run_add


def _center_from_mask(mask_np: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def run_replace(
    args,
    image_pil: Image.Image,
    mask_np: np.ndarray,
    src_name: str,
    tgt_name: str,
    device,
    dtype,
    tgt_mask_np: Optional[np.ndarray] = None,
    source_image_pil: Optional[Image.Image] = None,
) -> Image.Image:
    """
    replace = remove(src) -> add(tgt)

    要求：
      - 不管 tgt 在图中是否存在，tgt 都要放到 src 的位置（src_center）
    """
    src_center = _center_from_mask(mask_np)

    removed_img = run_remove(
        args=args,
        image_pil=image_pil,
        mask_np=mask_np,
        obj_name=src_name,
        device=device,
        dtype=dtype,
    )

    out_img = run_add(
        args=args,
        image_pil=removed_img,
        ref_mask_np=tgt_mask_np,
        source_image_pil=(source_image_pil or image_pil),
        target_center=src_center,  # 关键：永远用 src 的位置
    )
    return out_img
