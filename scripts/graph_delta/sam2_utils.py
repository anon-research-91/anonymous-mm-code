# scripts/graph_delta/sam2_utils.py
import os
import json
import numpy as np
from PIL import Image
import torch

from graph.mask_utils import clamp_bbox_xyxy

try:
    import cv2
except Exception:
    cv2 = None


def _resolve_sam2_cfg(cfg_path: str) -> str:
    if not cfg_path:
        return "configs/sam2/sam2_hiera_t.yaml"
    p = cfg_path.replace("\\", "/")
    if p.startswith("configs/"):
        return p
    idx = p.find("configs/")
    if idx != -1:
        return p[idx:]
    return p


def _prepare_sam2_import(sam2_repo_dir: str) -> None:
    sam2_repo_dir = os.path.abspath(sam2_repo_dir)
    if not os.path.exists(sam2_repo_dir):
        raise FileNotFoundError(f"sam2_repo_dir not found: {sam2_repo_dir}")
    import sys
    if sam2_repo_dir not in sys.path:
        sys.path.insert(0, sam2_repo_dir)
    os.chdir(sam2_repo_dir)


def build_sam2_predictor(sam2_repo_dir: str, sam2_cfg: str, sam2_ckpt: str, device: torch.device):
    _prepare_sam2_import(sam2_repo_dir)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        raise RuntimeError(
            "SAM2 import failed. Check --sam2_repo_dir points to the cloned repo (sam2/ and configs/). "
            f"Error: {e}"
        ) from e

    cfg_name = _resolve_sam2_cfg(sam2_cfg)
    sam2_ckpt = os.path.abspath(sam2_ckpt)
    if not os.path.exists(sam2_ckpt):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt}")
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    model = build_sam2(
        config_file=cfg_name,
        ckpt_path=sam2_ckpt,
        device=dev_str,
    )
    return SAM2ImagePredictor(model)


def ensure_vg_mask_for_obj(
    *,
    graph: dict,
    graph_path: str,
    image_pil: Image.Image,
    obj_id: int,
    predictor,
    out_mask_dir: str,
) -> str:
    nodes = {int(n["id"]): n for n in graph.get("nodes", []) if "id" in n}
    node = nodes.get(int(obj_id))
    if node is None:
        raise ValueError(f"obj_id={obj_id} not found in nodes")

    mask_rel = node.get("mask_path", None)
    if mask_rel and os.path.exists(mask_rel):
        return mask_rel

    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for SAM2 mask generation.")

    W, H = graph.get("image_size", [None, None])
    if W is None or H is None:
        W, H = image_pil.size

    bbox = node.get("bbox", None)
    if bbox is None:
        raise ValueError(f"node {obj_id} missing bbox")
    bbox = clamp_bbox_xyxy([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], int(W), int(H))

    os.makedirs(out_mask_dir, exist_ok=True)
    mask_abs = os.path.join(out_mask_dir, f"{graph.get('image_id','img')}_obj{obj_id}_mask.npy")

    img_rgb = np.array(image_pil.convert("RGB"))
    predictor.set_image(img_rgb)
    sam_box = np.array([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], dtype=np.float32)
    masks, _, _ = predictor.predict(box=sam_box, multimask_output=False)
    mask = masks[0]
    mask_u8 = (mask > 0).astype(np.uint8)
    np.save(mask_abs, mask_u8)

    for n in graph.get("nodes", []):
        if int(n.get("id", -1)) == int(obj_id):
            n["mask_path"] = mask_abs
            break

    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)
    return mask_abs
