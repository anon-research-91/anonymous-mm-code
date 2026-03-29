# scripts/graph_delta/pipeline_attribute.py
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from .sd_inpaint import SDInpaintEditor
from .mask_ops import refine_mask_binary, make_inpaint_mask_pil

COLOR_TO_HUE179 = {
    "red": 0, "orange": 15, "yellow": 30, "green": 60, "cyan": 90, "blue": 120,
    "purple": 145, "pink": 165, "brown": 15, "beige": 20, "gold": 25,
    "silver": 0, "gray": 0, "grey": 0, "black": 0, "white": 0,
}

MATERIAL_KEYWORDS = [
    "denim", "jean", "leather", "suede", "silk", "wool", "cotton", "linen",
    "metal", "latex", "plastic", "velvet", "fur", "knit", "satin", "canvas",
    "pattern", "textured", "texture", "fabric", "material", "embroidered",
    "checkered", "striped", "polka", "camouflage"
]


def is_complex_material_edit(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k in p for k in MATERIAL_KEYWORDS)


def _alpha_from_mask_distance(mask_bin: np.ndarray, feather: int = 7) -> np.ndarray:
    """
    alpha=1 inside object, only boundary transitions.
    Uses distanceTransform so interior stays 1.0 (removes 'overlay' look).
    """
    if cv2 is None:
        return (mask_bin > 0).astype(np.float32)

    m = (mask_bin > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    if feather <= 0:
        return (m > 0).astype(np.float32)

    alpha = np.clip(dist / float(feather), 0.0, 1.0)
    alpha[dist >= float(feather)] = 1.0
    return alpha.astype(np.float32)


def recolor_hsv_keep_texture(
    image_pil: Image.Image,
    mask_np: np.ndarray,
    target_color: str,
    feather: int = 7,
    dilate: int = 2,
    erode: int = 0,
    sat_scale: float = 1.15,
    min_sat: int = 35,
    protect_highlights: bool = True,
    highlight_v_thresh: int = 235,
    protect_shadows: bool = True,
    shadow_v_thresh: int = 25,
) -> Image.Image:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for HSV recolor.")
    target_color = (target_color or "").lower().strip()
    if target_color not in COLOR_TO_HUE179:
        return image_pil

    img = np.array(image_pil.convert("RGB"))

    mask_bin = refine_mask_binary(mask_np, dilate=dilate, erode=erode).astype(np.float32)
    m = (mask_bin > 0)
    if m.sum() == 0:
        return image_pil

    alpha = _alpha_from_mask_distance(mask_bin, feather=max(0, int(feather)))
    alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)

    mm = m.copy()
    if protect_highlights:
        mm = mm & (v < int(highlight_v_thresh))
    if protect_shadows:
        mm = mm & (v > int(shadow_v_thresh))

    tc = target_color
    h_new = h.copy()
    s_new = s.copy()
    v_new = v.copy()

    tgt_h = int(COLOR_TO_HUE179[tc])

    if tc in ["gray", "grey", "silver"]:
        s_new[mm] = np.minimum(s_new[mm], 25)
    elif tc == "white":
        s_new[mm] = np.minimum(s_new[mm], 12)
        v_new[mm] = np.maximum(v_new[mm], 235)
    elif tc == "black":
        s_new[mm] = np.minimum(s_new[mm], 18)
        v_new[mm] = np.minimum(v_new[mm], 70)
    else:
        h_new[mm] = tgt_h
        s_scaled = (s[mm].astype(np.float32) * float(sat_scale)).clip(0, 255).astype(np.int32)
        s_new[mm] = np.maximum(s_scaled, int(min_sat))
        v_new[mm] = v[mm]

    hsv_new = hsv.copy()
    hsv_new[:, :, 0] = h_new.astype(np.uint8)
    hsv_new[:, :, 1] = s_new.astype(np.uint8)
    hsv_new[:, :, 2] = v_new.astype(np.uint8)

    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB).astype(np.float32)
    rgb_old = img.astype(np.float32)

    out = (rgb_new * alpha3 + rgb_old * (1.0 - alpha3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def run_attribute(
    *,
    image_pil: Image.Image,
    mask_np: np.ndarray,
    tgt_cls: str,
    new_val: str,
    prompt_tgt: str,

    # route control
    attr_route: str = "auto",  # ["auto","recolor","inpaint"]

    # recolor params
    mask_feather: int = 7,
    hsv_sat_scale: float = 1.15,
    hsv_min_sat: int = 35,
    mask_dilate: int = 2,
    mask_erode: int = 0,

    # inpaint params
    sd_inpaint_model: str = "/root/models/sdxl-inpaint",
    device=None,
    dtype=None,
    steps: int = 30,
    guidance_scale: float = 5.5,
    seed: int = 123,
    negative_prompt: str = "blurry, low quality, artifacts, unrealistic",
    inpaint_dilate: int = 10,
    inpaint_blur: int = 13,
    attr_inpaint_strength: float = 0.65,
) -> Image.Image:
    """
    Attribute edit that does NOT depend on args.
    - If simple color and not complex material: HSV recolor keeps texture.
    - Else: SD inpaint.
    """
    new_val = (new_val or "").lower().strip()
    simple_color = (new_val in COLOR_TO_HUE179)
    complex_edit = is_complex_material_edit(prompt_tgt)

    route = attr_route
    if route == "auto":
        route = "inpaint" if (not simple_color) or complex_edit else "recolor"

    if route == "recolor":
        return recolor_hsv_keep_texture(
            image_pil=image_pil,
            mask_np=mask_np,
            target_color=new_val,
            feather=int(mask_feather),
            sat_scale=float(hsv_sat_scale),
            min_sat=int(hsv_min_sat),
            dilate=int(mask_dilate),
            erode=int(mask_erode),
        )

    # inpaint route
    if device is None:
        raise ValueError("run_attribute(inpaint) requires device")
    if dtype is None:
        raise ValueError("run_attribute(inpaint) requires dtype")

    sd_editor = SDInpaintEditor(model_id=sd_inpaint_model, device=device, torch_dtype=dtype)
    mask_pil = make_inpaint_mask_pil(
        mask_np,
        dilate=int(inpaint_dilate),
        blur=int(inpaint_blur),
        erode=int(mask_erode),
    )

    prompt_sd = (
        "realistic photo of the same scene. "
        f"inside the masked region, change the color of the {tgt_cls} to {new_val}. "
        "keep the original material texture and lighting. "
        "do not change anything outside the mask. no logo, no text."
    )
    neg_sd = (negative_prompt or "") + ", low detail, flat texture, oversmooth, painted look, plastic texture, deformed"

    return sd_editor.run_inpaint(
        image_pil=image_pil,
        mask_pil=mask_pil,
        prompt=prompt_sd,
        negative_prompt=neg_sd,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        strength=float(attr_inpaint_strength),
    )
