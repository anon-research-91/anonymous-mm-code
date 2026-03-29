# scripts/graph_delta/ablation_cli.py
from __future__ import annotations

import os
import re
import sys
import argparse
import subprocess
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image

from graph.graph_tokens import load_scene_graph
from scripts.graph_delta.io_utils import find_image_path_for_graph, load_mask_any
from scripts.graph_delta.parse_intent import parse_intent

from scripts.graph_delta.sd_inpaint import make_sd_inpaint_editor
from scripts.graph_delta.mask_ops import make_inpaint_mask_pil

from diffusion.instruct_pix2pix_editor import InstructPix2PixEditor, InstructEditConfig


# -------------------------
# Debug: save mask + overlay
# -------------------------
def save_mask_debug(out_dir: str, image_pil: Image.Image, mask_np, tag: str = "mask"):
    """
    Robust: accept mask as np array (or torch tensor) and write:
      - {tag}.png
      - {tag}_overlay.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # convert mask to numpy HxW
    if torch.is_tensor(mask_np):
        mask_np = mask_np.detach().float().cpu().numpy()

    mask_np = np.array(mask_np)
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"mask_np must be 2D HxW, got shape={mask_np.shape}")

    m = (mask_np > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(m, mode="L")
    mask_path = os.path.join(out_dir, f"{tag}.png")
    mask_pil.save(mask_path)

    img = np.array(image_pil.convert("RGB")).astype(np.uint8)
    overlay = img.copy()
    overlay[m > 0] = (overlay[m > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay, mode="RGB")
    overlay_path = os.path.join(out_dir, f"{tag}_overlay.png")
    overlay_pil.save(overlay_path)

    print(f"[DEBUG] saved mask -> {mask_path}", flush=True)
    print(f"[DEBUG] saved overlay -> {overlay_path}", flush=True)

    # list directory to prove it's written
    try:
        files = sorted(os.listdir(out_dir))
        print(f"[DEBUG] out_dir files: {files}", flush=True)
    except Exception as e:
        print(f"[DEBUG] listdir failed: {e}", flush=True)


# -------------------------
# Helpers: selection
# -------------------------
def _pick_class_from_prompt(prompt: str, class_vocab: List[str]) -> str:
    p = (prompt or "").lower()
    vocab = sorted({(c or "").lower().strip() for c in class_vocab if c}, key=len, reverse=True)
    for c in vocab:
        if c and re.search(rf"\b{re.escape(c)}\b", p):
            return c
    return ""


def _bbox_area(node: dict) -> float:
    b = node.get("bbox", None)
    if not b or len(b) < 4:
        return -1.0
    x0, y0, x1, y1 = map(float, b[:4])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_center(node: dict) -> Tuple[float, float]:
    b = node.get("bbox", None)
    if not b or len(b) < 4:
        return 0.0, 0.0
    x0, y0, x1, y1 = map(float, b[:4])
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _extract_spatial_from_prompt(text: str) -> str:
    t = (text or "").lower()
    if "leftmost" in t or "on the left" in t or "to the left" in t:
        return "left"
    if "rightmost" in t or "on the right" in t or "to the right" in t:
        return "right"
    if "topmost" in t or "at the top" in t:
        return "top"
    if "bottommost" in t or "at the bottom" in t:
        return "bottom"
    m = re.search(r"\b(left|right|top|bottom)\b", t)
    return m.group(1) if m else ""


def _fallback_select_of_class_with_spatial(graph: dict, cls: str, spatial: str = "") -> Optional[int]:
    cls = (cls or "").lower().strip()
    spatial = (spatial or "").lower().strip()
    nodes = graph.get("nodes", [])
    cand = [n for n in nodes if (n.get("class", "").lower().strip() == cls)]
    if not cand:
        return None

    if spatial:
        centers = [(n, *_bbox_center(n)) for n in cand]
        if spatial == "left":
            cand = [min(centers, key=lambda t: t[1])[0]]
        elif spatial == "right":
            cand = [max(centers, key=lambda t: t[1])[0]]
        elif spatial == "top":
            cand = [min(centers, key=lambda t: t[2])[0]]
        elif spatial == "bottom":
            cand = [max(centers, key=lambda t: t[2])[0]]

    pick = max(cand, key=_bbox_area)
    return int(pick["id"])


def _union_masks(a: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    if b is None:
        return (a > 0).astype(np.float32)
    return ((a > 0) | (b > 0)).astype(np.float32)


def _select_obj_id_for_intent(graph: dict, prompt: str, intent: str, info: dict) -> Optional[int]:
    intent = (intent or "").lower().strip()
    info = info or {}
    class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]

    if intent == "remove":
        cls = str(info.get("obj", "")).lower().strip() or _pick_class_from_prompt(prompt, class_vocab)
        spatial = str(info.get("obj_spatial", "") or info.get("spatial", "")).lower().strip()
        spatial = spatial or _extract_spatial_from_prompt(prompt)
        return _fallback_select_of_class_with_spatial(graph, cls, spatial)

    if intent == "replace":
        cls = str(info.get("src", "")).lower().strip() or _pick_class_from_prompt(prompt, class_vocab)
        spatial = str(info.get("src_spatial", "") or info.get("spatial", "")).lower().strip()
        spatial = spatial or _extract_spatial_from_prompt(prompt)
        return _fallback_select_of_class_with_spatial(graph, cls, spatial)

    if intent == "attribute":
        cls = str(info.get("obj", "")).lower().strip() or _pick_class_from_prompt(prompt, class_vocab)
        spatial = str(info.get("obj_spatial", "") or info.get("spatial", "")).lower().strip()
        spatial = spatial or _extract_spatial_from_prompt(prompt)
        return _fallback_select_of_class_with_spatial(graph, cls, spatial)

    if intent == "relation":
        cls = str(info.get("subj", "")).lower().strip() or _pick_class_from_prompt(prompt, class_vocab)
        subj_spatial = str(info.get("subj_spatial", "")).lower().strip()
        return _fallback_select_of_class_with_spatial(graph, cls, subj_spatial)

    if intent == "add":
        ref = str(info.get("ref", "")).lower().strip()
        ref_spatial = str(info.get("ref_spatial", "")).lower().strip()
        if ref:
            rid = _fallback_select_of_class_with_spatial(graph, ref, ref_spatial)
            if rid is not None:
                return rid
        cls = str(info.get("tgt", "")).lower().strip() or _pick_class_from_prompt(prompt, class_vocab)
        return _fallback_select_of_class_with_spatial(graph, cls, "")

    cls = _pick_class_from_prompt(prompt, class_vocab)
    if not cls:
        return None
    return _fallback_select_of_class_with_spatial(graph, cls, "")


# -------------------------
# Editors
# -------------------------
def run_ip2p_whole(args, image_pil: Image.Image, device: torch.device, dtype: torch.dtype) -> Image.Image:
    editor = InstructPix2PixEditor(model_id=args.ip2p_model, device=device, torch_dtype=dtype)
    cfg = InstructEditConfig(
        num_inference_steps=int(args.ip2p_steps),
        guidance_scale=float(args.ip2p_guidance_scale),
        image_guidance_scale=float(args.ip2p_image_guidance_scale),
        seed=int(args.seed),
    )
    return editor.run_edit(
        image_pil=image_pil,
        instruction=args.prompt_tgt,
        cfg=cfg,
        mask=None,
        graph=None,
        deltas=None,
        graph_lambda=0.0,
    )


def run_sdxl_inpaint_with_mask(
    args,
    image_pil: Image.Image,
    mask_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
) -> Image.Image:
    mask_pil = make_inpaint_mask_pil(
        mask_np,
        dilate=int(args.inpaint_dilate),
        blur=int(args.inpaint_blur),
        erode=int(args.mask_erode),
    )

    sd = make_sd_inpaint_editor(
        model_id=str(args.sd_inpaint_model),
        device=device,
        torch_dtype=dtype,
        fallback_model_id=getattr(args, "sd_inpaint_fallback_model", None),
        prefer_safetensors=True,
    )

    neg = (args.negative_prompt or "") + ", text, logo, watermark, frame, border, extra object, blurry, artifacts"

    out = sd.run_inpaint(
        image_pil=image_pil,
        mask_pil=mask_pil,
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        strength=float(args.unified_strength),
    )
    return out


# -------------------------
# Ablations
# -------------------------
def run_wo_type_aware_unified_inpaint(
    args,
    image_pil: Image.Image,
    graph: dict,
    graph_path: str,
    device: torch.device,
    dtype: torch.dtype,
    intent: str,
    info: dict,
) -> Image.Image:
    W, H = image_pil.size
    intent_l = (intent or "").lower().strip()

    mask_np: Optional[np.ndarray] = None

    if intent_l == "relation":
        class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]
        subj_cls = str(info.get("subj", "")).lower().strip() or _pick_class_from_prompt(args.prompt_tgt, class_vocab)
        obj_cls = str(info.get("obj", "")).lower().strip() or _pick_class_from_prompt(args.prompt_tgt, class_vocab)
        sid = _fallback_select_of_class_with_spatial(graph, subj_cls, str(info.get("subj_spatial", "")).lower().strip())
        oid = _fallback_select_of_class_with_spatial(graph, obj_cls, str(info.get("obj_spatial", "")).lower().strip())
        if sid is not None:
            m1 = load_mask_any(graph, int(sid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)
            m2 = None
            if oid is not None:
                m2 = load_mask_any(graph, int(oid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)
            mask_np = _union_masks(np.array(m1), np.array(m2) if m2 is not None else None)

    if mask_np is None:
        oid = _select_obj_id_for_intent(graph, args.prompt_tgt, intent, info)
        if oid is None:
            mask_np = np.ones((H, W), dtype=np.float32)
        else:
            mask_np = load_mask_any(graph, int(oid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)

    # HARD debug proof
    if getattr(args, "debug_mask", False):
        mn = float(np.min(mask_np)) if not torch.is_tensor(mask_np) else float(mask_np.min().item())
        mx = float(np.max(mask_np)) if not torch.is_tensor(mask_np) else float(mask_np.max().item())
        shp = getattr(mask_np, "shape", None)
        print(f"[DEBUG] debug_mask=True | out_dir={args._debug_out_dir}", flush=True)
        print(f"[DEBUG] mask shape={shp} min={mn} max={mx} intent={intent_l}", flush=True)
        save_mask_debug(out_dir=args._debug_out_dir, image_pil=image_pil, mask_np=mask_np, tag="selected_mask")

    prompt = (
        "realistic photo of the same scene. "
        f"Inside the masked region, follow the instruction: {args.prompt_tgt}. "
        "Keep everything outside the mask unchanged. No text, no watermark."
    )

    return run_sdxl_inpaint_with_mask(
        args=args,
        image_pil=image_pil,
        mask_np=np.array(mask_np),
        device=device,
        dtype=dtype,
        prompt=prompt,
    )


def run_wo_region_split_sdxl_mask_ones(
    args,
    image_pil: Image.Image,
    graph: dict,
    graph_path: str,
    device: torch.device,
    dtype: torch.dtype,
    intent: str,
    info: dict,
) -> Image.Image:
    W, H = image_pil.size
    intent_l = (intent or "").lower().strip()

    # debug-only selection
    selected_mask_np: Optional[np.ndarray] = None
    if intent_l == "relation":
        class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]
        subj_cls = str(info.get("subj", "")).lower().strip() or _pick_class_from_prompt(args.prompt_tgt, class_vocab)
        obj_cls = str(info.get("obj", "")).lower().strip() or _pick_class_from_prompt(args.prompt_tgt, class_vocab)
        sid = _fallback_select_of_class_with_spatial(graph, subj_cls, str(info.get("subj_spatial", "")).lower().strip())
        oid = _fallback_select_of_class_with_spatial(graph, obj_cls, str(info.get("obj_spatial", "")).lower().strip())
        if sid is not None:
            m1 = load_mask_any(graph, int(sid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)
            m2 = None
            if oid is not None:
                m2 = load_mask_any(graph, int(oid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)
            selected_mask_np = _union_masks(np.array(m1), np.array(m2) if m2 is not None else None)
    else:
        oid = _select_obj_id_for_intent(graph, args.prompt_tgt, intent, info)
        if oid is not None:
            selected_mask_np = load_mask_any(graph, int(oid), args.mask_mode, graph_path=graph_path, yolo_sam_root=args.yolo_sam_root)

    if getattr(args, "debug_mask", False) and selected_mask_np is not None:
        print(f"[DEBUG] debug_mask=True | out_dir={args._debug_out_dir} | intent={intent_l}", flush=True)
        save_mask_debug(out_dir=args._debug_out_dir, image_pil=image_pil, mask_np=selected_mask_np, tag="selected_mask")

    # execution mask=ones
    mask_ones = np.ones((H, W), dtype=np.float32)

    prompt = (
        "realistic photo of the same scene. "
        f"Follow the instruction: {args.prompt_tgt}. "
        "No text, no watermark."
    )

    return run_sdxl_inpaint_with_mask(
        args=args,
        image_pil=image_pil,
        mask_np=mask_ones,
        device=device,
        dtype=dtype,
        prompt=prompt,
    )


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "wo_structure", "wo_type_aware", "wo_region_split"],
    )

    p.add_argument("--image_id", type=str, required=True)
    p.add_argument("--prompt_tgt", type=str, required=True)

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--yolo_sam_root", type=str, default="scripts/yolo_sam2_outputs")
    p.add_argument("--graph_root", type=str, default="/root/SGGE_DM/output/scene_graphs_vg")
    p.add_argument("--image_search_root", type=str, default=None)
    p.add_argument("--mask_mode", type=str, default="mask", choices=["bbox", "mask", "none"])

    p.add_argument("--sd_inpaint_model", type=str, default="/root/models/sdxl-inpaint")
    p.add_argument("--sd_inpaint_fallback_model", type=str, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, artifacts, unrealistic")
    p.add_argument("--unified_strength", type=float, default=0.85)

    p.add_argument("--inpaint_dilate", type=int, default=40)
    p.add_argument("--inpaint_blur", type=int, default=18)
    p.add_argument("--mask_erode", type=int, default=0)

    p.add_argument("--ip2p_model", type=str, default="/root/SGGE_DM/instruct-pix2pix")
    p.add_argument("--ip2p_steps", type=int, default=40)
    p.add_argument("--ip2p_guidance_scale", type=float, default=5.0)
    p.add_argument("--ip2p_image_guidance_scale", type=float, default=1.8)

    p.add_argument("--debug_mask", action="store_true")

    args, unknown = p.parse_known_args()

    device = torch.device("cuda") if (torch.cuda.is_available() and not args.cpu) else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    out_dir = args.out_dir or f"/root/SGGE_DM/output/ablation/{args.ablation}/{args.image_id}"
    os.makedirs(out_dir, exist_ok=True)
    args._debug_out_dir = out_dir

    graph_path = os.path.join(args.graph_root, f"{args.image_id}_scene_graph.json")
    graph = load_scene_graph(graph_path)

    img_path = find_image_path_for_graph(graph, graph_path, args.image_id, search_root=args.image_search_root)
    if not img_path:
        raise FileNotFoundError(f"image not found for image_id={args.image_id}")
    image_pil = Image.open(img_path).convert("RGB")

    if args.ablation == "full":
        cmd = [
            sys.executable, "-m", "scripts.graph_delta.cli",
            "--image_id", args.image_id,
            "--prompt_tgt", args.prompt_tgt,
            "--graph_root", args.graph_root,
            "--yolo_sam_root", args.yolo_sam_root,
            "--mask_mode", args.mask_mode,
            "--sd_inpaint_model", args.sd_inpaint_model,
            "--steps", str(args.steps),
            "--guidance_scale", str(args.guidance_scale),
            "--seed", str(args.seed),
            "--negative_prompt", args.negative_prompt,
            "--inpaint_dilate", str(args.inpaint_dilate),
            "--inpaint_blur", str(args.inpaint_blur),
            "--mask_erode", str(args.mask_erode),
        ]
        if args.sd_inpaint_fallback_model:
            cmd += ["--sd_inpaint_fallback_model", args.sd_inpaint_fallback_model]
        if args.image_search_root:
            cmd += ["--image_search_root", args.image_search_root]
        if args.cpu:
            cmd += ["--cpu"]
        if unknown:
            cmd += unknown
        subprocess.run(cmd, check=True)
        print("[OK] full pipeline finished (saved by scripts/graph_delta/cli.py)", flush=True)
        return

    intent, info = parse_intent(args.prompt_tgt)
    info = dict(info or {})

    if args.ablation == "wo_structure":
        out = run_ip2p_whole(args, image_pil=image_pil, device=device, dtype=dtype)
        out_path = os.path.join(out_dir, "edited.png")
        out.save(out_path)
        print(f"[OK] saved -> {out_path}", flush=True)
        return

    if args.ablation == "wo_type_aware":
        out = run_wo_type_aware_unified_inpaint(
            args=args,
            image_pil=image_pil,
            graph=graph,
            graph_path=graph_path,
            device=device,
            dtype=dtype,
            intent=intent,
            info=info,
        )
        out_path = os.path.join(out_dir, "edited.png")
        out.save(out_path)
        print(f"[OK] saved -> {out_path}", flush=True)
        return

    if args.ablation == "wo_region_split":
        out = run_wo_region_split_sdxl_mask_ones(
            args=args,
            image_pil=image_pil,
            graph=graph,
            graph_path=graph_path,
            device=device,
            dtype=dtype,
            intent=intent,
            info=info,
        )
        out_path = os.path.join(out_dir, "edited.png")
        out.save(out_path)
        print(f"[OK] saved -> {out_path}", flush=True)
        return

    raise ValueError(args.ablation)


if __name__ == "__main__":
    main()
