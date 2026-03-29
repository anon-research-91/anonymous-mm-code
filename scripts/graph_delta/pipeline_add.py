# scripts/graph_delta/pipeline_add.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from .sd_inpaint import SDInpaintEditor, make_sd_inpaint_editor
from .mask_ops import (
    make_inpaint_mask_pil,
    make_ring_mask,
    guess_target_category,
    add_mask_params_for_target,
    make_add_mask_for_replace_bbox_clamped,
    bbox_from_mask_np,
    align_bbox_to_multiple,
    fill_mask_region_with_bg_noise,
)
from .prompts import build_add_prompt, build_add_negative


def _default_add_bbox(W: int, H: int):
    # 桌面 top-down：中间偏下比较稳
    x0 = int(W * 0.35)
    x1 = int(W * 0.65)
    y0 = int(H * 0.45)
    y1 = int(H * 0.80)
    return (x0, y0, x1, y1)


def _bbox_from_center(cx: int, cy: int, bw: int, bh: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = int(cx - bw // 2)
    y0 = int(cy - bh // 2)
    x1 = x0 + int(bw)
    y1 = y0 + int(bh)

    # clamp with shift (keep size)
    dx0 = max(0, -x0)
    dy0 = max(0, -y0)
    dx1 = max(0, x1 - W)
    dy1 = max(0, y1 - H)

    x0 += dx0 - dx1
    x1 += dx0 - dx1
    y0 += dy0 - dy1
    y1 += dy0 - dy1

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    return (x0, y0, x1, y1)


def _alpha_feather_rgba(patch_rgba: Image.Image, feather: int = 2) -> Image.Image:
    if feather <= 0 or cv2 is None:
        return patch_rgba
    arr = np.array(patch_rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return patch_rgba
    a = arr[:, :, 3]
    k = feather * 2 + 1
    arr[:, :, 3] = cv2.GaussianBlur(a, (k, k), 0)
    return Image.fromarray(arr, mode="RGBA")


def _bbox_from_mask_np01(mask_np01: np.ndarray, expand: int = 2) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_np01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0 -= expand
    y0 -= expand
    x1 += expand
    y1 += expand
    return (x0, y0, x1, y1)


def extract_object_rgba(
    image_pil: Image.Image, mask_np01: np.ndarray, expand: int = 2
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    W, H = image_pil.size
    x0, y0, x1, y1 = _bbox_from_mask_np01(mask_np01, expand=expand)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    base_rgba = image_pil.convert("RGBA")
    mask_img = Image.fromarray((mask_np01 * 255).astype(np.uint8), mode="L").resize((W, H), Image.NEAREST)

    obj_full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    obj_full.paste(base_rgba, (0, 0), mask_img)
    return obj_full.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def paste_object_to_target(
    base_rgb: Image.Image,
    obj_rgba: Image.Image,
    target_center: Tuple[int, int],
    use_seamless: bool = True,
    feather: int = 2,
    seamless_mode: str = "NORMAL",
) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns:
      moved_img (RGB)
      moved_mask_np (H,W float32 0/1) at NEW position
    """
    W, H = base_rgb.size
    obj_rgba = _alpha_feather_rgba(obj_rgba, feather=feather)
    w, h = obj_rgba.size
    cx, cy = target_center
    tx0 = int(max(0, min(W - w, cx - w // 2)))
    ty0 = int(max(0, min(H - h, cy - h // 2)))

    obj_a = np.array(obj_rgba)[:, :, 3]
    obj_mask = (obj_a > 10).astype(np.float32)

    moved_mask = np.zeros((H, W), dtype=np.float32)
    moved_mask[ty0 : ty0 + h, tx0 : tx0 + w] = np.maximum(
        moved_mask[ty0 : ty0 + h, tx0 : tx0 + w], obj_mask
    )

    if use_seamless and cv2 is not None:
        base = np.array(base_rgb.convert("RGB"))
        obj = np.array(obj_rgba.convert("RGBA"))
        obj_rgb = obj[:, :, :3]
        mask_u8 = (obj[:, :, 3] > 10).astype(np.uint8) * 255
        center = (tx0 + w // 2, ty0 + h // 2)

        mode = cv2.NORMAL_CLONE
        if str(seamless_mode).upper().strip() == "MIXED":
            mode = cv2.MIXED_CLONE

        try:
            blended = cv2.seamlessClone(obj_rgb, base, mask_u8, center, mode)
            return Image.fromarray(blended, mode="RGB"), moved_mask
        except Exception:
            pass

    out = base_rgb.convert("RGBA")
    out.alpha_composite(obj_rgba, (tx0, ty0))
    return out.convert("RGB"), moved_mask


def _make_ring_mask(mask_np01: np.ndarray, outer_k: int = 19, inner_k: int = 9) -> np.ndarray:
    m = (mask_np01 > 0).astype(np.uint8)
    if cv2 is None:
        return (mask_np01 > 0).astype(np.float32)

    ok = int(outer_k) | 1
    ik = int(inner_k) | 1
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ik, ik))
    dil = cv2.dilate(m, k1, iterations=1)
    ero = cv2.erode(m, k2, iterations=1)
    ring = np.clip(dil.astype(np.int16) - ero.astype(np.int16), 0, 1).astype(np.float32)
    return ring


def remove_extra_instances_keep_largest(
    sd_editor,
    before_add_img: Image.Image,
    after_add_img: Image.Image,
    crop_bbox: Tuple[int, int, int, int],
    negative_prompt: str,
    seed: int,
) -> Image.Image:
    """
    这是“SD 画出来的 add”去重用的；copy-paste add 不要调用它（不然会把你新增的副本抹掉）
    """
    if cv2 is None:
        return after_add_img

    x0, y0, x1, y1 = crop_bbox
    a0 = np.array(before_add_img.convert("RGB"))[y0:y1, x0:x1].astype(np.int16)
    a1 = np.array(after_add_img.convert("RGB"))[y0:y1, x0:x1].astype(np.int16)
    diff = np.abs(a1 - a0).mean(axis=2).astype(np.uint8)

    m = (diff > 18).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 2:
        return after_add_img

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    remove_mask = (labels != 0) & (labels != keep)
    if remove_mask.sum() < 40:
        return after_add_img

    H, W = np.array(after_add_img.convert("RGB")).shape[:2]
    full = np.zeros((H, W), dtype=np.uint8)
    full[y0:y1, x0:x1][remove_mask] = 255

    mask_pil = make_inpaint_mask_pil((full > 0).astype(np.float32), dilate=6, blur=11, erode=0)
    prompt = (
        "realistic photo. remove the extra duplicated objects in the masked region. "
        "keep only one object. reconstruct the wooden table background naturally. "
        "do not add new objects."
    )
    cleaned = sd_editor.run_inpaint(
        image_pil=after_add_img,
        mask_pil=mask_pil,
        prompt=prompt,
        negative_prompt=(negative_prompt or "") + ", extra objects, duplicates",
        num_inference_steps=24,
        guidance_scale=4.5,
        seed=seed + 333,
        strength=0.55,
    )
    return cleaned


def run_add(
    args,
    image_pil: Image.Image,
    ref_mask_np: np.ndarray = None,
    source_image_pil: Optional[Image.Image] = None,
    target_center: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    add:
      - if ref_mask_np is provided: copy-paste an existing instance from source_image_pil
      - else: SD draw

    target_center:
      - if provided: BOTH branches should respect it
      - if not: fallback to default add bbox center / default add bbox
    """
    W, H = image_pil.size
    source_image_pil = source_image_pil or image_pil

    # 读目标类别
    args._intent_info = getattr(args, "_intent_info", {}) or {}
    tgt_name = (args._intent_info.get("tgt") or args._intent_info.get("obj") or args.prompt_tgt or "object").lower().strip()
    tgt_category = guess_target_category(tgt_name)

    # ------------------------------------------------
    # A) copy-paste add（图里已有该物体）
    # ------------------------------------------------
    if ref_mask_np is not None:
        if target_center is None:
            x0, y0, x1, y1 = _default_add_bbox(W, H)
            target_center = ((x0 + x1) // 2, (y0 + y1) // 2)

        ref_mask01 = (ref_mask_np > 0).astype(np.float32)
        obj_rgba, _ = extract_object_rgba(source_image_pil, ref_mask01, expand=2)

        use_seamless = bool(getattr(args, "move_seamless", False))
        seamless_mode = str(getattr(args, "move_seamless_mode", "NORMAL"))
        feather = int(getattr(args, "move_alpha_feather", 2))

        moved, moved_mask_np = paste_object_to_target(
            base_rgb=image_pil.convert("RGB"),
            obj_rgba=obj_rgba,
            target_center=target_center,
            use_seamless=use_seamless,
            feather=feather,
            seamless_mode=seamless_mode,
        )

        sd = make_sd_inpaint_editor(
            model_id=str(getattr(args, "sd_inpaint_model", "")),
            device=getattr(args, "_device", "cuda"),
            torch_dtype=getattr(args, "_dtype", None),
            fallback_model_id=getattr(args, "sd_inpaint_fallback_model", None),
            prefer_safetensors=True,
        )

        ring_np = _make_ring_mask(moved_mask_np, outer_k=19, inner_k=9)
        fusion_mask = make_inpaint_mask_pil(
            ring_np,
            dilate=int(getattr(args, "fusion_dilate", 2)),
            blur=int(getattr(args, "fusion_blur", 9)),
            erode=0,
        )

        neg_user = (getattr(args, "negative_prompt", "") or "").strip()
        neg = (neg_user + ", " if neg_user else "") + "artifacts, ghosting, smear, blurry, extra objects"

        fusion_prompt = (
            "realistic photo. "
            "blend edges naturally into the scene. "
            "add subtle contact shadows on the table near the object boundary. "
            "preserve background texture and sharpness. "
            "do not change object shape. do not add new objects."
        )

        seed = int(getattr(args, "seed", 123))
        final = sd.run_inpaint(
            image_pil=moved,
            mask_pil=fusion_mask,
            prompt=fusion_prompt,
            negative_prompt=neg,
            num_inference_steps=int(getattr(args, "fusion_steps", 24)),
            guidance_scale=float(getattr(args, "fusion_guidance_scale", 4.0)),
            seed=seed + 199,
            strength=float(getattr(args, "fusion_strength", 0.22)),
        )
        return final

    # ------------------------------------------------
    # B) SD draw add（图里没有该物体） —— 支持 target_center
    # ------------------------------------------------
    sd_editor = SDInpaintEditor(model_id=args.sd_inpaint_model, device=args._device, torch_dtype=args._dtype)

    # 用 default bbox 作为模板尺寸
    dx0, dy0, dx1, dy1 = _default_add_bbox(W, H)
    bw, bh = (dx1 - dx0), (dy1 - dy0)

    if target_center is not None:
        cx, cy = target_center
        x0, y0, x1, y1 = _bbox_from_center(int(cx), int(cy), bw, bh, W, H)
    else:
        x0, y0, x1, y1 = dx0, dy0, dx1, dy1

    mask_add_bin = np.zeros((H, W), dtype=np.float32)
    mask_add_bin[y0:y1, x0:x1] = 1.0
    mask_add_hard_pil = Image.fromarray((mask_add_bin * 255).astype(np.uint8), mode="L")
    crop_bbox = (x0, y0, x1, y1)

    # 2) SD mask：dilate 一点，但 blur=0
    mask_add = make_inpaint_mask_pil(
        mask_add_bin,
        dilate=int(getattr(args, "replace_add_dilate", 8)),
        blur=0,
        erode=int(getattr(args, "mask_erode", 0)),
    )

    # 3) noise init（按 replace 思路）
    add_init = fill_mask_region_with_bg_noise(
        image_pil,
        mask_add_hard_pil,
        seed=args.seed + 101,
        noise_std=14,
        blur=5,
    )

    prompt_add = build_add_prompt(tgt_name, tgt_category)
    neg_add = build_add_negative(tgt_name, src="", category=tgt_category, base_neg=args.negative_prompt or "")

    added = sd_editor.run_inpaint_crop(
        image_pil=add_init,
        mask_pil=mask_add,
        prompt=prompt_add,
        negative_prompt=neg_add,
        num_inference_steps=max(int(getattr(args, "steps", 30)), 50),
        guidance_scale=float(getattr(args, "replace_guidance", 5.0)),
        seed=args.seed + 1,
        strength=float(getattr(args, "replace_strength", 0.85)),
        crop_bbox=crop_bbox,
        blend_blur=int(getattr(args, "crop_blend_blur", 9)),
    )

    # 4) ring refine
    ring_mask = make_ring_mask(mask_add_bin, dilate=7, erode=2, blur=7)
    refined = sd_editor.run_inpaint_crop(
        image_pil=added,
        mask_pil=ring_mask,
        prompt=(
            f"realistic photo. keep the {tgt_name} exactly. refine edges and blending. "
            "match lighting and preserve sharp background texture. do not add new objects."
        ),
        negative_prompt=(args.negative_prompt or "") + ", extra objects, artifacts, blurry",
        num_inference_steps=22,
        guidance_scale=4.5,
        seed=args.seed + 9,
        strength=0.12,
        crop_bbox=crop_bbox,
        blend_blur=7,
    )

    # 5) 去重（用原图当 baseline）
    refined = remove_extra_instances_keep_largest(
        sd_editor=sd_editor,
        before_add_img=image_pil,
        after_add_img=refined,
        crop_bbox=crop_bbox,
        negative_prompt=(args.negative_prompt or ""),
        seed=args.seed + 9,
    )
    return refined
