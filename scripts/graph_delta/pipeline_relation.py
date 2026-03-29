# scripts/graph_delta/pipeline_relation.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from .sd_inpaint import make_sd_inpaint_editor
from .mask_ops import make_inpaint_mask_pil, get_bbox_from_mask

# ✅ 复用 add 的 extract/paste（与 replace/add 效果一致）
from .pipeline_add import extract_object_rgba as extract_object_rgba_add
from .pipeline_add import paste_object_to_target as paste_object_to_target_add

# ✅ 关键：relation 删除阶段直接复用 remove pipeline
from .pipeline_remove import run_remove as run_remove_pipeline


def _get_out_dir(args) -> str:
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        return out_dir
    image_id = getattr(args, "image_id", "unknown")
    return f"/root/SGGE_DM/output/graph_delta_instruct_edits/{image_id}"


# 保留函数名：只输出 overlay（避免一堆 debug 图）
def _save_mask_debug(image_pil: Image.Image, mask_np: np.ndarray, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    img = np.array(image_pil.convert("RGB")).copy()
    mm = (mask_np > 0)

    img[mm, 0] = np.clip(img[mm, 0] * 0.35 + 200, 0, 255).astype(np.uint8)
    img[mm, 1] = np.clip(img[mm, 1] * 0.35, 0, 255).astype(np.uint8)
    img[mm, 2] = np.clip(img[mm, 2] * 0.35, 0, 255).astype(np.uint8)

    path = os.path.join(out_dir, f"{name}_mask_overlay.png")
    Image.fromarray(img, mode="RGB").save(path)
    print(f"[DEBUG] saved mask overlay -> {path}", flush=True)


def repair_thin_mask(mask_np: np.ndarray, close_k: int = 9, min_area: int = 60) -> np.ndarray:
    """
    修复细长物体（fork/knife）mask 断裂/缺一截：
      1) 二值闭运算连接断裂
      2) 只保留最大连通域
    """
    if cv2 is None:
        return (mask_np > 0).astype(np.float32)

    m = (mask_np > 0).astype(np.uint8)
    k = int(close_k) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m.astype(np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(1 + areas.argmax())
    if stats[idx, cv2.CC_STAT_AREA] < int(min_area):
        return m.astype(np.float32)

    return (labels == idx).astype(np.float32)


def compute_target_center_for_relation(
    rel: str,
    src_bbox: Tuple[int, int, int, int],
    ref_bbox: Optional[Tuple[int, int, int, int]],
    W: int,
    H: int,
) -> Tuple[int, int]:
    rel = (rel or "on").lower().strip()
    sx0, sy0, sx1, sy1 = src_bbox
    sw, sh = (sx1 - sx0), (sy1 - sy0)

    if ref_bbox is None:
        return (sx0 + sx1) // 2, (sy0 + sy1) // 2

    rx0, ry0, rx1, ry1 = ref_bbox
    rcx, rcy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    margin = max(8, int(0.06 * max(sw, sh)))

    if "left" in rel:
        return max(sw // 2 + 1, rx0 - margin - sw // 2), rcy
    if "right" in rel:
        return min(W - sw // 2 - 1, rx1 + margin + sw // 2), rcy
    if "above" in rel or "top" in rel:
        return rcx, max(sh // 2 + 1, ry0 - margin - sh // 2)
    if "below" in rel or "under" in rel or "bottom" in rel:
        return rcx, min(H - sh // 2 - 1, ry1 + margin + sh // 2)

    return rcx, rcy


def _make_ring_mask(mask_np_full: np.ndarray, out_px: int = 12, in_px: int = 4) -> np.ndarray:
    """
    只做边缘一圈的 mask（外膨胀 - 内腐蚀），避免整块区域被 inpaint 糊掉。
    """
    m = (mask_np_full > 0).astype(np.uint8)
    if cv2 is None:
        return m.astype(np.float32)

    out_px = int(max(1, out_px))
    in_px = int(max(0, in_px))

    k1 = (out_px * 2 + 1, out_px * 2 + 1)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k1)
    dil = cv2.dilate(m, ker1, iterations=1)

    if in_px > 0:
        k2 = (in_px * 2 + 1, in_px * 2 + 1)
        ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k2)
        ero = cv2.erode(m, ker2, iterations=1)
    else:
        ero = m

    ring = (dil.astype(np.int16) - ero.astype(np.int16))
    ring = (ring > 0).astype(np.float32)
    return ring


def _erode_mask(mask_np_full: np.ndarray, erode_px: int = 2) -> np.ndarray:
    m = (mask_np_full > 0).astype(np.uint8)
    if cv2 is None or erode_px <= 0:
        return m.astype(np.float32)
    k = (erode_px * 2 + 1, erode_px * 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
    e = cv2.erode(m, ker, iterations=1)
    return e.astype(np.float32)


def _composite_keep_object(
    fused_rgb: Image.Image,
    moved_rgb: Image.Image,
    placed_mask_full: np.ndarray,
    erode_px: int = 2,
) -> Image.Image:
    """
    把 moved 里“锐利的主体”覆盖回 fused，避免 fusion 把主体糊掉。
    只需要把 mask 稍微腐蚀一下，留边缘给 fused 的阴影/融合。
    """
    keep = _erode_mask(placed_mask_full, erode_px=erode_px)
    keep_u8 = (keep > 0).astype(np.uint8) * 255
    keep_pil = Image.fromarray(keep_u8, mode="L")

    a = fused_rgb.convert("RGB")
    b = moved_rgb.convert("RGB")
    out = Image.composite(b, a, keep_pil)  # mask=255 -> b
    return out


def run_relation(
    args,
    image_pil: Image.Image,
    mask_np_subj: np.ndarray,
    mask_np_obj: Optional[np.ndarray],
    rel: str,
    device,
    dtype,
) -> Image.Image:
    """
    relation:
      1) subj mask repair + debug overlay (SUBJ/OBJ)
      2) remove subj  ✅ now calls pipeline_remove.run_remove
      3) extract subj patch
      4) compute target
      5) paste -> moved + placed_mask(新位置)
      6) edge-ring fusion inpaint（只修边缘/阴影）
      7) composite keep sharp object（回贴主体，防糊/防曝光）
    """
    out_dir = _get_out_dir(args)

    subj_id = getattr(args, "_debug_subj_id", "unknown")
    obj_id2 = getattr(args, "_debug_obj_id2", "unknown")

    # --- masks debug + repair thin objects ---
    mask_np_subj = repair_thin_mask(mask_np_subj, close_k=9, min_area=60)
    _save_mask_debug(image_pil, mask_np_subj, out_dir, name=f"relation_SUBJ_{subj_id}")

    if mask_np_obj is not None:
        _save_mask_debug(image_pil, (mask_np_obj > 0).astype(np.float32), out_dir, name=f"relation_OBJ_{obj_id2}")

    # --- load editor (used later for fusion) ---
    sd = make_sd_inpaint_editor(
        model_id=str(getattr(args, "sd_inpaint_model", "")),
        device=device,
        torch_dtype=dtype,
        fallback_model_id=getattr(args, "sd_inpaint_fallback_model", None),
        prefer_safetensors=True,
    )

    # ============================================================
    # ✅ remove: call pipeline_remove.run_remove (same as remove intent)
    # ============================================================
    info = getattr(args, "_intent_info", {}) or {}
    subj_name = str(info.get("subj", "") or "object").lower().strip() or "object"

    removed = run_remove_pipeline(
        args=args,
        image_pil=image_pil,
        mask_np=mask_np_subj,
        obj_name=subj_name,
        device=device,
        dtype=dtype,
    )
    if removed.size != image_pil.size:
        removed = removed.resize(image_pil.size, Image.BICUBIC)

    # --- extract object patch from ORIGINAL image ---
    subj_mask01 = (mask_np_subj > 0).astype(np.float32)
    obj_rgba, src_bbox = extract_object_rgba_add(image_pil, subj_mask01, expand=2)

    # --- ref bbox from obj mask ---
    ref_bbox = None
    if mask_np_obj is not None:
        try:
            ref_bbox = get_bbox_from_mask(mask_np_obj, expand=2)
        except Exception:
            ref_bbox = None

    W, H = image_pil.size
    cx, cy = compute_target_center_for_relation(rel, src_bbox, ref_bbox, W, H)

    use_seamless = bool(getattr(args, "move_seamless", True))
    seamless_mode = str(getattr(args, "move_seamless_mode", "NORMAL"))
    feather = int(getattr(args, "move_alpha_feather", 2))

    # -----------------------------
    # ✅ FIX: keep a non-seamless copy for "keep sharp object"
    # moved_blend: for boundary naturalness (seamless clone)
    # moved_raw  : for restoring true object appearance (avoid over-exposure)
    # -----------------------------
    moved_raw, placed_mask = paste_object_to_target_add(
        base_rgb=removed.convert("RGB"),
        obj_rgba=obj_rgba,
        target_center=(cx, cy),
        use_seamless=False,  # always keep a raw version
        feather=feather,
        seamless_mode=seamless_mode,
    )

    if use_seamless:
        moved_blend, _ = paste_object_to_target_add(
            base_rgb=removed.convert("RGB"),
            obj_rgba=obj_rgba,
            target_center=(cx, cy),
            use_seamless=True,
            feather=feather,
            seamless_mode=seamless_mode,
        )
    else:
        moved_blend = moved_raw

    # =========================
    # edge-ring fusion inpaint
    # =========================
    seed = int(getattr(args, "seed", 123))

    ring_np = _make_ring_mask(placed_mask, out_px=12, in_px=4)

    fusion_mask = make_inpaint_mask_pil(
        ring_np,
        dilate=0,
        blur=9,
        erode=0,
    )

    neg_user = (getattr(args, "negative_prompt", "") or "").strip()
    neg = (neg_user + ", " if neg_user else "") + "artifacts, ghosting, smear, blurry, extra objects"

    fusion_prompt = (
        "realistic photo. "
        "blend the object edges naturally into the wooden table. "
        "add subtle contact shadows and match local lighting to the surroundings. "
        "keep the object sharp and do not change its shape. "
        "do not add any new objects."
    )

    fused = sd.run_inpaint(
        image_pil=moved_blend,
        mask_pil=fusion_mask,
        prompt=fusion_prompt,
        negative_prompt=neg,
        num_inference_steps=22,
        guidance_scale=4.0,
        seed=seed + 99,
        strength=0.18,
    )

    # ✅ IMPORTANT: restore object from moved_raw (not moved_blend)
    final = _composite_keep_object(
        fused_rgb=fused,
        moved_rgb=moved_raw,
        placed_mask_full=placed_mask,
        erode_px=2,
    )
    return final
