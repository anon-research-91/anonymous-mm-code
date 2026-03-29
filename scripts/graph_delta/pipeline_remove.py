# scripts/graph_delta/pipeline_remove.py
from __future__ import annotations

import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None

from .sd_inpaint import make_sd_inpaint_editor
from .mask_ops import make_inpaint_mask_pil


# ---------------------------
# helpers
# ---------------------------
def _get_out_dir(args) -> str:
    out_dir = getattr(args, "out_dir", None)
    if isinstance(out_dir, str) and out_dir.strip():
        return out_dir
    image_id = getattr(args, "image_id", "unknown")
    return f"/root/SGGE_DM/output/graph_delta_instruct_edits/{image_id}"


def _ival(v, default: int) -> int:
    if v is None:
        return int(default)
    return int(v)


def _fval(v, default: float) -> float:
    if v is None:
        return float(default)
    return float(v)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def _ensure_mask_l_same_size(mask_pil: Image.Image, image_pil: Image.Image) -> Image.Image:
    if mask_pil.mode != "L":
        mask_pil = mask_pil.convert("L")
    if mask_pil.size != image_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)
    return mask_pil


def repair_mask(mask_np: np.ndarray, close_k: int = 7, close_iter: int = 1, min_area: int = 40) -> np.ndarray:
    """mask cleanup: binarize + close + remove tiny comps"""
    m = (mask_np > 0).astype(np.uint8)
    if cv2 is None:
        return m.astype(np.float32)

    if close_k and close_k > 1 and close_iter and close_iter > 0:
        k = int(close_k) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num > 1:
            keep = np.zeros_like(m, dtype=np.uint8)
            for i in range(1, num):
                if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
                    keep[labels == i] = 1
            if keep.sum() > 0:
                m = keep

    return m.astype(np.float32)


def _obj_token(obj_name: str) -> str:
    obj = (obj_name or "").strip()
    return obj.replace("_", " ") if obj else ""


def _mask_np_from_pil(mask_pil: Image.Image) -> np.ndarray:
    m = np.array(mask_pil.convert("L"))
    return (m > 0).astype(np.float32)


def _mean_abs_diff(img_a: Image.Image, img_b: Image.Image, region_np: np.ndarray) -> float:
    r = (region_np > 0)
    if not r.any():
        return 0.0
    a = np.array(_ensure_rgb(img_a)).astype(np.float32)
    b = np.array(_ensure_rgb(img_b)).astype(np.float32)
    d = np.abs(a - b)
    return float(d[r].mean())


def _score_mask_choice(before: Image.Image, after: Image.Image, in_np: np.ndarray, out_np: np.ndarray, lam: float = 0.6) -> float:
    """
    score = inside_change - lam * outside_change
    避免选到“反 mask”导致外面大改、里面没改
    """
    inside = _mean_abs_diff(before, after, in_np)
    outside = _mean_abs_diff(before, after, out_np)
    return float(inside - lam * outside)


def _run_inpaint_try_both_masks(
    sd,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    ref_in_np: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    strength: float,
    tag: str = "",
) -> Image.Image:
    """
    mask polarity 选择：normal / inverted 二选一
    """
    img = _ensure_rgb(image_pil)
    m0 = _ensure_mask_l_same_size(mask_pil, img)
    m1 = ImageOps.invert(m0)

    in_np = (ref_in_np > 0).astype(np.float32)
    out_np = (1.0 - in_np).astype(np.float32)

    out0 = sd.run_inpaint(
        image_pil=img,
        mask_pil=m0,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        strength=float(strength),
    )
    out0 = _ensure_rgb(out0)
    if out0.size != img.size:
        out0 = out0.resize(img.size, Image.BICUBIC)
    s0 = _score_mask_choice(img, out0, in_np, out_np, lam=0.6)

    out1 = sd.run_inpaint(
        image_pil=img,
        mask_pil=m1,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed) + 1000,
        strength=float(strength),
    )
    out1 = _ensure_rgb(out1)
    if out1.size != img.size:
        out1 = out1.resize(img.size, Image.BICUBIC)
    s1 = _score_mask_choice(img, out1, in_np, out_np, lam=0.6)

    pick = "inverted" if s1 > s0 else "normal"
    print(f"[MASK:{tag}] score normal={s0:.3f} inverted={s1:.3f} -> pick {pick}", flush=True)
    return out1 if s1 > s0 else out0


def _cv_inpaint_base(image_pil: Image.Image, hard_mask_pil: Image.Image, method: str = "ns", radius: int = 11) -> Image.Image:
    if cv2 is None:
        return _ensure_rgb(image_pil)

    img = np.array(_ensure_rgb(image_pil))
    m = np.array(hard_mask_pil.convert("L"))
    m = (m > 0).astype(np.uint8) * 255

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_NS if (method or "").lower() in {"ns", "navier"} else cv2.INPAINT_TELEA
    r = int(max(1, min(radius, 30)))
    out = cv2.inpaint(bgr, m, r, flag)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _ring_mask_from_binary(bin_mask_np: np.ndarray, ring_width: int = 28) -> np.ndarray:
    ring_width = int(max(2, min(ring_width, 128)))
    m = (bin_mask_np > 0).astype(np.uint8)
    if cv2 is None:
        return m.astype(np.float32)

    k1 = (ring_width * 2 + 1) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    dil = cv2.dilate(m, kernel, iterations=1)

    k2 = (max(1, ring_width - 2) * 2 + 1) | 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    ero = cv2.erode(m, kernel2, iterations=1)

    ring = np.clip(dil - ero, 0, 1).astype(np.float32)
    return ring


def _replace_mask_with_noise(image_pil: Image.Image, hard_mask_np: np.ndarray, seed: int, noise_strength: float = 1.0) -> Image.Image:
    img = np.array(_ensure_rgb(image_pil)).astype(np.float32)
    m = (hard_mask_np > 0)
    if not m.any():
        return _ensure_rgb(image_pil)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    noise = rng.randn(*img.shape).astype(np.float32) * 255.0

    a = float(np.clip(noise_strength, 0.0, 1.0))
    for c in range(3):
        ch = img[:, :, c]
        nh = noise[:, :, c]
        ch[m] = (1.0 - a) * ch[m] + a * nh[m]
        img[:, :, c] = ch

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


def _feather_mask_from_hard(hard_mask_pil: Image.Image, feather: int = 6) -> Image.Image:
    feather = int(max(0, min(feather, 24)))
    m = hard_mask_pil.convert("L")
    if feather <= 0:
        return m
    if cv2 is None:
        return m.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    mn = (np.array(m) > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(mn, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mn, cv2.DIST_L2, 3)
    denom = (dist_in + dist_out + 1e-6)
    alpha = dist_in / denom
    alpha = np.clip(alpha ** (1.0 / max(1.0, feather / 4.0)), 0.0, 1.0)
    return Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")


def _composite_inside(original: Image.Image, edited: Image.Image, alpha_mask_l: Image.Image) -> Image.Image:
    a = np.array(alpha_mask_l.convert("L")).astype(np.float32) / 255.0
    a = np.clip(a, 0.0, 1.0)
    a3 = np.repeat(a[:, :, None], 3, axis=2)
    o = np.array(_ensure_rgb(original)).astype(np.float32)
    e = np.array(_ensure_rgb(edited)).astype(np.float32)
    out = o * (1.0 - a3) + e * a3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# --------- auto-tune + scoring (NEW) ---------
def _mask_area_ratio(mask_np01: np.ndarray) -> float:
    m = (mask_np01 > 0)
    if m.size == 0:
        return 0.0
    return float(m.mean())


def _edge_density_in_ring(image_pil: Image.Image, ring_np01: np.ndarray) -> float:
    """
    粗略判别边界附近纹理复杂度：
    - 地板木纹/地毯这种：ring 边缘区域边缘密度高
    - 天空/墙面这种：低
    """
    if cv2 is None:
        return 0.0
    r = (ring_np01 > 0)
    if not r.any():
        return 0.0
    g = cv2.cvtColor(np.array(_ensure_rgb(image_pil)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 60, 150)
    return float(edges[r].mean() / 255.0)


def _auto_params(area_ratio: float, edge_density: float):
    """
    返回一组更通用的默认参数（可被 args 覆盖）
    核心思想：
    - 规则/强纹理（edge_density 高）=> 更依赖 CV，SD 更克制（低 guidance/适中 strength）
    - 洞越大 => CV 半径、dilate、ring、feather 更大
    """
    # size bucket
    if area_ratio < 0.01:
        dilate = 22
        ring = 16
        feather = 5
        cv_r = 11
    elif area_ratio < 0.05:
        dilate = 32
        ring = 20
        feather = 6
        cv_r = 13
    else:
        dilate = 44
        ring = 28
        feather = 8
        cv_r = 15

    # texture bucket
    if edge_density >= 0.08:
        # 强纹理：SD 别太“画”
        destroy_guidance = 1.35
        destroy_strength = 0.88
        repair_strength = 0.10
        sd_blur = 1
        cv_method = "ns"
    else:
        destroy_guidance = 1.8
        destroy_strength = 0.92
        repair_strength = 0.14
        sd_blur = 2
        cv_method = "ns"

    return dict(
        dilate=dilate,
        ring_width=ring,
        feather=feather,
        cv_radius=cv_r,
        cv_method=cv_method,
        destroy_guidance=destroy_guidance,
        destroy_strength=destroy_strength,
        repair_strength=repair_strength,
        sd_blur=sd_blur,
    )


def _seam_score(original: Image.Image, out_img: Image.Image, hard_mask_np01: np.ndarray) -> float:
    """
    候选择优评分：越大越好
    目标：边界连续、纹理统计接近
    """
    if cv2 is None:
        return 0.0
    m = (hard_mask_np01 > 0).astype(np.uint8)
    if m.sum() < 10:
        return 0.0

    g0 = cv2.cvtColor(np.array(_ensure_rgb(original)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(np.array(_ensure_rgb(out_img)), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # boundary band: dilate - erode
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dil = cv2.dilate(m, k, iterations=1)
    ero = cv2.erode(m, k, iterations=1)
    band = (dil - ero) > 0

    # inside/outside narrow band split
    inside_band = band & (m > 0)
    outside_band = band & (m == 0)
    if inside_band.sum() < 20 or outside_band.sum() < 20:
        return 0.0

    # boundary continuity
    diff = float(np.abs(g1[inside_band].mean() - g1[outside_band].mean()))
    # texture variance match
    var_in = float(g1[inside_band].var())
    var_out = float(g1[outside_band].var())
    var_gap = abs(var_in - var_out)

    # penalize unnatural smoothing: compare with original outside variance
    var_out0 = float(g0[outside_band].var())
    smooth_pen = abs(var_out - var_out0)

    # higher score is better
    return float(-diff - 0.002 * var_gap - 0.002 * smooth_pen)


# ---------------------------
# Prompts
# ---------------------------
def build_fill_prompt(obj_name: str) -> str:
    # 不在正向 prompt 里提具体 obj 名，避免“画回来”
    return (
        "a realistic photo of the same scene. "
        "Remove the object in the masked region. "
        "Fill the masked region by extending the existing background naturally. "
        "Match local texture, lighting, shadows and perspective. "
        "No object, no vehicle, no subject, no silhouette, no structure. "
        "No hard edges, no recognizable outline. "
        "Do not change anything outside the mask."
    )


def build_repair_prompt() -> str:
    return (
        "a realistic photo of the same scene. "
        "Blend the masked edge seamlessly into surrounding textures. "
        "Only extend existing textures and lighting. "
        "Remove seams, halos, and artifacts. "
        "Do not introduce any object, subject, or structure. "
        "Do not change anything outside the mask."
    )


def build_negative_prompt(user_neg: str = "", obj_name: str = "") -> str:
    obj = _obj_token(obj_name)
    obj_block = f"{obj}, " if obj else ""

    vehicle_block = (
        "car, vehicle, automobile, sedan, suv, hatchback, truck, van, bus, taxi, "
        "motorcycle, scooter, bicycle, "
        "wheel, tire, rim, axle, bumper, headlight, taillight, license plate, "
        "windshield, window, door, handle, mirror, grille, hood, trunk, chassis, "
        "vehicle silhouette, car silhouette, outline of a car, "
    )

    anti_entity = (
        "foreground subject, main subject, focal object, salient object, "
        "standalone object, isolated object, distinct object, extra object, added object, inserted object, "
        "new object, new subject, new foreground, "
        "recognizable object, recognizable shape, recognizable structure, "
        "person, human, animal, "
        "hard edge, sharp outline, high contrast edges, cutout look, "
    )

    base = (
        obj_block
        + vehicle_block
        + anti_entity
        + "text, letters, numbers, logo, watermark, sign, symbol, "
        + "artifacts, unrealistic, painting, artwork, illustration, poster, frame, border"
    )

    user_neg = (user_neg or "").strip()
    return (user_neg + ", " if user_neg else "") + base


# ---------------------------
# stages
# ---------------------------
def sd_fill_full_hole(sd, init_img: Image.Image, fill_mask_pil: Image.Image, hard_ref_np: np.ndarray, obj_name: str, seed: int,
                     num_inference_steps: int, guidance_scale: float, strength: float, negative_prompt: str) -> Image.Image:
    prompt = build_fill_prompt(obj_name)
    return _run_inpaint_try_both_masks(
        sd=sd,
        image_pil=_ensure_rgb(init_img),
        mask_pil=fill_mask_pil,
        ref_in_np=hard_ref_np,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        strength=float(strength),
        tag="fill",
    )


def sd_repair_ring(sd, base_img: Image.Image, ring_mask_pil: Image.Image, ring_ref_np: np.ndarray, obj_name: str, seed: int,
                   num_inference_steps: int, guidance_scale: float, strength: float, negative_prompt: str) -> Image.Image:
    prompt = build_repair_prompt()
    return _run_inpaint_try_both_masks(
        sd=sd,
        image_pil=_ensure_rgb(base_img),
        mask_pil=ring_mask_pil,
        ref_in_np=ring_ref_np,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        strength=float(strength),
        tag="repair-ring",
    )


# ---------------------------
# main
# ---------------------------
def run_remove(
    args,
    image_pil: Image.Image,
    mask_np: np.ndarray,
    obj_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    out_dir = _get_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    base_seed = int(getattr(args, "seed", 123))

    # 1) clean ref mask
    ref_mask_np = repair_mask(
        mask_np,
        close_k=_ival(getattr(args, "close_k", None), 7),
        close_iter=_ival(getattr(args, "close_iter", None), 1),
        min_area=_ival(getattr(args, "min_area", None), 40),
    )
    ref_mask_np = (ref_mask_np > 0).astype(np.float32)

    # 2) sd editor
    sd = make_sd_inpaint_editor(
        model_id=str(getattr(args, "sd_inpaint_model", "")),
        device=device,
        torch_dtype=dtype,
        fallback_model_id=getattr(args, "sd_inpaint_fallback_model", None),
        prefer_safetensors=True,
    )

    # multi-trial for better generalization
    num_trials = _ival(getattr(args, "remove_trials", None), 3)  # NEW: 默认 3 次
    num_trials = max(1, min(num_trials, 8))
    auto_tune = bool(getattr(args, "auto_tune_remove", True))    # NEW: 默认开启

    best_img = _ensure_rgb(image_pil)
    best_score = -1e9

    # pre-neg prompt
    neg = build_negative_prompt(getattr(args, "negative_prompt", ""), obj_name=obj_name)

    for ti in range(num_trials):
        seed = base_seed + ti * 97

        # ---- choose params (auto or args) ----
        # a) start from args
        dilate = _ival(getattr(args, "inpaint_dilate", None), 40)
        erode = _ival(getattr(args, "mask_erode", None), 0)
        ring_width = _ival(getattr(args, "ring_width", None), 28)
        feather = _ival(getattr(args, "composite_feather", None), 6)
        cv_radius = _ival(getattr(args, "cv_inpaint_radius", None), 11)
        cv_method = str(getattr(args, "cv_inpaint_method", "ns"))  # FIX: use args

        destroy_steps = _ival(getattr(args, "destroy_steps", None), 40)
        destroy_guidance = _fval(getattr(args, "destroy_guidance_scale", None), 1.8)
        destroy_strength = _fval(getattr(args, "destroy_strength", None), 0.90)
        noise_strength = _fval(getattr(args, "destroy_noise_strength", None), 1.0)

        repair_steps = _ival(getattr(args, "repair_steps", None), 22)
        repair_guidance = _fval(getattr(args, "repair_guidance_scale", None), 2.0)
        repair_strength = _fval(getattr(args, "repair_strength", None), 0.14)

        # b) auto-tune override (if enabled)
        # build a temporary hard mask first for auto stats
        hard_mask_tmp = make_inpaint_mask_pil(ref_mask_np, dilate=max(6, dilate // 2), blur=0, erode=0)
        hard_mask_tmp = _ensure_mask_l_same_size(hard_mask_tmp, image_pil)
        hard_np_tmp = _mask_np_from_pil(hard_mask_tmp)
        ring_np_tmp = _ring_mask_from_binary(hard_np_tmp, ring_width=ring_width)
        area_ratio = _mask_area_ratio(hard_np_tmp)
        edge_den = _edge_density_in_ring(image_pil, ring_np_tmp)

        if auto_tune:
            ap = _auto_params(area_ratio, edge_den)
            dilate = int(getattr(args, "inpaint_dilate", ap["dilate"]))
            ring_width = int(getattr(args, "ring_width", ap["ring_width"]))
            feather = int(getattr(args, "composite_feather", ap["feather"]))
            cv_radius = int(getattr(args, "cv_inpaint_radius", ap["cv_radius"]))
            cv_method = str(getattr(args, "cv_inpaint_method", ap["cv_method"]))

            destroy_guidance = float(getattr(args, "destroy_guidance_scale", ap["destroy_guidance"]))
            destroy_strength = float(getattr(args, "destroy_strength", ap["destroy_strength"]))
            repair_strength = float(getattr(args, "repair_strength", ap["repair_strength"]))

            sd_blur = int(getattr(args, "inpaint_blur", ap["sd_blur"]))
        else:
            sd_blur = _ival(getattr(args, "inpaint_blur", None), 2)

        # clamp
        dilate = max(0, min(dilate, 128))
        erode = max(0, min(erode, 64))
        ring_width = max(2, min(ring_width, 128))
        feather = max(0, min(feather, 24))
        cv_radius = max(1, min(cv_radius, 30))
        sd_blur = max(0, min(sd_blur, 8))

        destroy_steps = max(18, min(destroy_steps, 80))
        destroy_guidance = float(np.clip(destroy_guidance, 1.0, 2.6))
        destroy_strength = float(np.clip(destroy_strength, 0.80, 0.98))
        noise_strength = float(np.clip(noise_strength, 0.0, 1.0))

        repair_steps = max(8, min(repair_steps, 60))
        repair_guidance = float(np.clip(repair_guidance, 1.0, 4.0))
        repair_strength = float(np.clip(repair_strength, 0.06, 0.28))

        # ---- build masks with tuned params ----
        fill_mask_pil = make_inpaint_mask_pil(ref_mask_np, dilate=dilate, blur=sd_blur, erode=erode)
        fill_mask_pil = _ensure_mask_l_same_size(fill_mask_pil, image_pil)

        hard_mask_pil = make_inpaint_mask_pil(ref_mask_np, dilate=max(6, dilate // 2), blur=0, erode=0)
        hard_mask_pil = _ensure_mask_l_same_size(hard_mask_pil, image_pil)
        hard_mask_np = _mask_np_from_pil(hard_mask_pil)

        comp_mask_pil = _feather_mask_from_hard(hard_mask_pil, feather=feather)
        comp_mask_pil = _ensure_mask_l_same_size(comp_mask_pil, image_pil)

        ring_np = _ring_mask_from_binary(hard_mask_np, ring_width=ring_width)
        ring_mask_pil = make_inpaint_mask_pil(ring_np, dilate=0, blur=4, erode=0)
        ring_mask_pil = _ensure_mask_l_same_size(ring_mask_pil, image_pil)

        # ---- stage A: CV base ----
        img_cv = _cv_inpaint_base(image_pil, hard_mask_pil, method=cv_method, radius=cv_radius)

        # ---- stage B: SD fill (noise init) ----
        init_for_sd = _replace_mask_with_noise(img_cv, hard_mask_np, seed=seed, noise_strength=noise_strength)

        img_fill_raw = sd_fill_full_hole(
            sd=sd,
            init_img=init_for_sd,
            fill_mask_pil=fill_mask_pil,
            hard_ref_np=hard_mask_np,
            obj_name=obj_name,
            seed=seed,
            num_inference_steps=destroy_steps,
            guidance_scale=destroy_guidance,
            strength=destroy_strength,
            negative_prompt=neg,
        )
        if img_fill_raw.size != image_pil.size:
            img_fill_raw = img_fill_raw.resize(image_pil.size, Image.BICUBIC)

        img_fill = _composite_inside(original=image_pil, edited=img_fill_raw, alpha_mask_l=comp_mask_pil)

        # ---- stage C: ring repair ----
        if bool(getattr(args, "enable_cleanup", True)):
            img_ring_raw = sd_repair_ring(
                sd=sd,
                base_img=img_fill,
                ring_mask_pil=ring_mask_pil,
                ring_ref_np=ring_np,
                obj_name=obj_name,
                seed=seed + 7,
                num_inference_steps=repair_steps,
                guidance_scale=repair_guidance,
                strength=repair_strength,
                negative_prompt=neg,
            )
            if img_ring_raw.size != image_pil.size:
                img_ring_raw = img_ring_raw.resize(image_pil.size, Image.BICUBIC)

            ring_alpha = ring_mask_pil.convert("L")
            img_out = _composite_inside(original=img_fill, edited=img_ring_raw, alpha_mask_l=ring_alpha)
        else:
            img_out = img_fill

        # ---- pick best by seam score ----
        sc = _seam_score(image_pil, img_out, hard_mask_np)
        print(
            f"[REMOVE][trial={ti}] area={area_ratio:.4f} edge={edge_den:.4f} "
            f"dilate={dilate} ring={ring_width} feather={feather} cv={cv_method}:{cv_radius} "
            f"destroy(gs={destroy_guidance:.2f},st={destroy_strength:.2f}) repair(st={repair_strength:.2f}) score={sc:.4f}",
            flush=True,
        )

        if sc > best_score:
            best_score = sc
            best_img = img_out

    return best_img
