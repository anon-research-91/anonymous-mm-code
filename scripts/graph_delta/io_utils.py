# scripts/graph_delta/io_utils.py
import os
import json
import subprocess
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from graph.mask_utils import clamp_bbox_xyxy, bbox_to_mask_xyxy


def save_no_resize(img: Image.Image, path: str, size: Optional[Tuple[int, int]] = None):
    """
    Avoid unnecessary resize to prevent blur.
    - size=None: save directly
    - size=(W,H): resize only if mismatch
    """
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.BICUBIC)
    img.save(path)


def find_image_path_for_graph(graph, image_id, image_search_root=None):

    # 永远强制从 test/no_edit 读取
    fixed_path = f"./test/no_edit/{image_id}.jpg"

    if not os.path.exists(fixed_path):
        raise FileNotFoundError(
            f"Image not found: {fixed_path}"
        )

    return fixed_path

def _infer_image_id_from_graph_path(graph_path: Optional[str]) -> str:
    if not graph_path:
        return "unknown"
    base = os.path.basename(graph_path)
    # expected: 63_scene_graph.json
    if base.endswith("_scene_graph.json"):
        return base.replace("_scene_graph.json", "")
    # fallback: take stem
    return os.path.splitext(base)[0]


def _resolve_mask_path(mask_rel: str, graph_path: Optional[str], yolo_sam_root: str) -> Optional[str]:
    """
    Resolve node.mask_path if it exists.
    Keep old behavior, but don't assume it exists.
    """
    if not mask_rel:
        return None
    if os.path.isabs(mask_rel) and os.path.exists(mask_rel):
        return mask_rel
    if graph_path:
        gp = os.path.join(os.path.dirname(graph_path), mask_rel)
        if os.path.exists(gp):
            return gp
    yp = os.path.join(yolo_sam_root, mask_rel)
    if os.path.exists(yp):
        return yp
    if os.path.exists(mask_rel):
        return mask_rel
    return None


def _guess_masks_vg_paths(
    image_id: str,
    obj_id: int,
    masks_vg_root: str = "./test/masks_vg",
) -> list:
    """
    Your masks are like:
      /root/SGGE_DM/output/masks_vg/63/63_obj1533910_mask.npy   (foldered)
    or sometimes:
      /root/SGGE_DM/output/masks_vg/63_obj1533910_mask.npy      (flat)
    """
    image_id = str(image_id)
    obj_id = int(obj_id)
    fn1 = f"{image_id}_obj{obj_id}_mask.npy"
    fn2 = f"{image_id}_obj{obj_id}.npy"  # just in case
    fn3 = f"{image_id}_obj{obj_id}_mask.png"  # (rare) for debug, not used

    cand = [
        os.path.join(masks_vg_root, image_id, fn1),
        os.path.join(masks_vg_root, fn1),
        os.path.join(masks_vg_root, image_id, fn2),
        os.path.join(masks_vg_root, fn2),
        os.path.join(masks_vg_root, image_id, fn3),
        os.path.join(masks_vg_root, fn3),
    ]
    return cand


def _load_mask_file(path: str, H: int, W: int) -> np.ndarray:
    """
    Load .npy mask and normalize to float32 0/1 (H,W).
    """
    m = np.load(path)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.float32)
    if m.shape[0] != H or m.shape[1] != W:
        # be strict: resize with nearest
        from PIL import Image as _PILImage
        mp = _PILImage.fromarray((m * 255).astype(np.uint8), mode="L").resize((W, H), resample=_PILImage.NEAREST)
        m = (np.array(mp) > 0).astype(np.float32)
    return m


def _try_autogen_mask_with_sam2(
    graph_path: str,
    masks_out_dir: str,
    obj_id: int,
) -> bool:
    """
    Call scripts/vg/vg_scene_graph_to_sam2_masks.py to generate missing mask for this obj_id.
    This will also write back mask_path into graph json (default behavior of that script).
    """
    if not graph_path or (not os.path.exists(graph_path)):
        return False

    # locate repo root (SGGE_DM/)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gen_script = os.path.join(repo_root, "scripts", "vg", "vg_scene_graph_to_sam2_masks.py")
    if not os.path.exists(gen_script):
        print(f"[MASK][AUTOGEN][WARN] generator script not found: {gen_script}", flush=True)
        return False

    os.makedirs(masks_out_dir, exist_ok=True)

    cmd = [
        "python",
        gen_script,
        "--graph_json",
        graph_path,
        "--masks_out_dir",
        masks_out_dir,
        "--only_obj_ids",
        str(int(obj_id)),
    ]

    print(f"[MASK][AUTOGEN] running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"[MASK][AUTOGEN][WARN] failed: {e}", flush=True)
        return False


def load_mask_any(
    graph: dict,
    obj_id: int,
    mask_mode: str,
    graph_path: Optional[str] = None,
    yolo_sam_root: str = "",
) -> np.ndarray:
    """
    mask_mode:
      - none: all ones
      - mask: prefer fitted mask (node.mask_path or /root/SGGE_DM/output/masks_vg search; if missing auto-generate via SAM2)
      - bbox: bbox -> mask
    """
    nodes = {int(n["id"]): n for n in graph.get("nodes", []) if "id" in n}
    node = nodes.get(int(obj_id))
    if node is None:
        raise ValueError(f"obj_id={obj_id} not found in graph nodes")

    W, H = graph.get("image_size", [None, None])
    if W is None or H is None:
        raise ValueError("graph missing image_size=[W,H]")
    W, H = int(W), int(H)

    if mask_mode == "none":
        return np.ones((H, W), dtype=np.float32)

    # -------------------------
    # MASK MODE: prefer fitted mask
    # -------------------------
    if mask_mode == "mask":
        # 1) try node.mask_path
        mask_rel = node.get("mask_path", None)
        if mask_rel:
            mask_path = _resolve_mask_path(mask_rel, graph_path, yolo_sam_root)
            if mask_path and os.path.exists(mask_path):
                return _load_mask_file(mask_path, H=H, W=W)

        # 2) search in /root/SGGE_DM/output/masks_vg (foldered + flat)
        image_id = _infer_image_id_from_graph_path(graph_path)
        for p in _guess_masks_vg_paths(image_id=image_id, obj_id=int(obj_id)):
            if p and os.path.exists(p) and p.endswith(".npy"):
                print(f"[MASK] resolved from masks_vg -> {p}", flush=True)
                # write back into node & graph json (so next time no miss)
                try:
                    node["mask_path"] = p
                    if graph_path and os.path.exists(graph_path):
                        # update the original json in-place
                        # NOTE: nodes dict is a copy; we need to update in graph["nodes"]
                        for n in graph.get("nodes", []):
                            if int(n.get("id", -1)) == int(obj_id):
                                n["mask_path"] = p
                                break
                        with open(graph_path, "w") as f:
                            json.dump(graph, f, indent=2)
                        print(f"[MASK] wrote back mask_path into graph -> {graph_path}", flush=True)
                except Exception as e:
                    print(f"[MASK][WARN] failed to write back mask_path: {e}", flush=True)

                return _load_mask_file(p, H=H, W=W)

        # 3) auto-generate with SAM2 into /root/SGGE_DM/output/masks_vg/<image_id>/
        masks_vg_root = "./test/masks_vg"
        masks_out_dir = os.path.join(masks_vg_root, str(image_id))
        ok = _try_autogen_mask_with_sam2(
            graph_path=str(graph_path) if graph_path else "",
            masks_out_dir=masks_out_dir,
            obj_id=int(obj_id),
        )
        if ok:
            # after autogen, try resolve again (prefer newly written mask_path in json)
            # reload graph json to get updated node.mask_path
            try:
                if graph_path and os.path.exists(graph_path):
                    with open(graph_path, "r") as f:
                        graph2 = json.load(f)
                    return load_mask_any(graph2, obj_id=int(obj_id), mask_mode="mask", graph_path=graph_path, yolo_sam_root=yolo_sam_root)
            except Exception as e:
                print(f"[MASK][AUTOGEN][WARN] reload after autogen failed: {e}", flush=True)

        # 4) last resort: fallback to bbox (do NOT crash)
        print(f"[MASK][WARN] fitted mask missing for obj_id={obj_id}, fallback to bbox.", flush=True)
        mask_mode = "bbox"

    # -------------------------
    # BBOX MODE
    # -------------------------
    bbox = node.get("bbox", None)
    if bbox is None:
        raise ValueError(f"node {obj_id} missing bbox")

    bbox = clamp_bbox_xyxy([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], int(W), int(H))
    return bbox_to_mask_xyxy(bbox, int(W), int(H))
