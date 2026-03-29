# scripts/graph_delta/cli.py
from __future__ import annotations

import os
import re
import argparse
from typing import Dict, Optional, Tuple, List

import torch
from PIL import Image

from graph.graph_tokens import load_scene_graph

from scripts.graph_delta.io_utils import find_image_path_for_graph, load_mask_any
from scripts.graph_delta.parse_intent import parse_intent
from scripts.graph_delta.compat import setup_tmpdir
from scripts.graph_delta.pipeline_attribute import run_attribute
from scripts.graph_delta.pipeline_remove import run_remove
from scripts.graph_delta.pipeline_replace import run_replace
from scripts.graph_delta.pipeline_relation import run_relation, compute_target_center_for_relation
from scripts.graph_delta.pipeline_add import run_add
from scripts.graph_delta.mask_ops import get_bbox_from_mask

try:
    from scripts.graph_delta.selector import select_object_id as _select_object_id
except Exception:
    _select_object_id = None


# -------------------------
# small utils
# -------------------------
def _print(msg: str):
    print(msg, flush=True)


def _nodes_map(graph: dict) -> Dict[int, dict]:
    return {int(n["id"]): n for n in graph.get("nodes", []) if "id" in n}


def _bbox_center(node: dict) -> Tuple[float, float]:
    b = node.get("bbox", None)
    if not b or len(b) < 4:
        return 0.0, 0.0
    x0, y0, x1, y1 = map(float, b[:4])
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_area(node: dict) -> float:
    b = node.get("bbox", None)
    if not b or len(b) < 4:
        return -1.0
    x0, y0, x1, y1 = map(float, b[:4])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _all_classes(graph: dict) -> List[str]:
    classes = sorted(
        {
            (n.get("class", "") or "").lower().strip()
            for n in graph.get("nodes", [])
            if n.get("class")
        }
    )
    return [c for c in classes if c]


def _format_classes(graph: dict) -> str:
    cls = _all_classes(graph)
    return f"available classes ({len(cls)}): " + (", ".join(cls) if cls else "(none)")


def _target_spec(cls: str, spatial: str = "") -> dict:
    return {
        "class": (cls or "").lower().strip(),
        "spatial": (spatial or "").lower().strip(),
        "appearance": {},
    }


def _pick_class_from_prompt(prompt: str, class_vocab) -> str:
    p = (prompt or "").lower()
    vocab = sorted({(c or "").lower().strip() for c in class_vocab if c}, key=len, reverse=True)
    for c in vocab:
        if c and re.search(rf"\b{re.escape(c)}\b", p):
            return c
    return ""


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


def _get_spatial_from_info(info: dict, keys) -> str:
    for k in keys:
        v = info.get(k, None)
        if v:
            return str(v).lower().strip()
    return ""


# -------------------------
# selection (strict by class)
# -------------------------
def _fallback_select_object_id_strict(graph: dict, target: dict) -> Optional[int]:
    nodes = graph.get("nodes", [])
    if not nodes:
        return None

    tgt_cls = (target.get("class") or "").lower().strip()
    spatial = (target.get("spatial") or "").lower().strip()
    if not tgt_cls:
        return None

    cand = [n for n in nodes if (n.get("class", "").lower().strip() == tgt_cls)]
    if not cand:
        return None

    if spatial:
        centers = [(n, *_bbox_center(n)) for n in cand]
        spatial = spatial.replace("most", "").strip()
        spatial = spatial.replace("right-hand", "right").replace("left-hand", "left")
        spatial = spatial.replace("upper", "top").replace("lower", "bottom")

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


def select_object_id(
    graph: dict,
    target: dict,
    device,
    clip_local_path: Optional[str] = None,
    strict: bool = True,
) -> Optional[int]:
    # 优先用外部 selector（如果项目里有）
    if _select_object_id is not None:
        try:
            oid = int(_select_object_id(graph, target, device=device, clip_local_path=clip_local_path))
        except TypeError:
            oid = int(_select_object_id(graph, target, device=device))
        except Exception:
            oid = None

        if oid is not None and strict:
            tgt_cls = (target.get("class") or "").lower().strip()
            if tgt_cls:
                n = _nodes_map(graph).get(int(oid), {})
                if (n.get("class", "").lower().strip() != tgt_cls):
                    return None
        if oid is not None:
            return int(oid)

    return _fallback_select_object_id_strict(graph, target)


def require_object_id(
    graph: dict,
    target: dict,
    device,
    clip_local_path: Optional[str],
    err_tag: str,
) -> int:
    oid = select_object_id(graph, target, device=device, clip_local_path=clip_local_path, strict=True)
    if oid is None:
        tgt_cls = (target.get("class") or "").lower().strip()
        spatial = (target.get("spatial") or "").lower().strip()
        _print(f"[ERROR:{err_tag}] class not found: '{tgt_cls}' spatial='{spatial}'")
        _print(f"[ERROR:{err_tag}] {_format_classes(graph)}")
        raise LookupError(f"{err_tag}: class not found: {tgt_cls}")
    return int(oid)


def _debug_pick(graph: dict, obj_id: int, tag: str):
    n = _nodes_map(graph).get(int(obj_id), {})
    _print(f"[PICK:{tag}] id={obj_id} class={n.get('class')} bbox={n.get('bbox')}")


# -------------------------
# add prompt parsers
# -------------------------
def _parse_add_relation(prompt: str):
    # add spoon left of right knife
    p = (prompt or "").lower().strip()
    p = re.sub(r"^add\s+(a|an|the)\s+", "add ", p)
    m = re.match(
        r"^add\s+(?P<tgt>\w+)\s+(?P<rel>left of|right of|above|below)\s+(?:(?P<spatial>left|right|top|bottom)\s+)?(?P<ref>\w+)\s*$",
        p,
    )
    if not m:
        return None
    return (
        m.group("tgt"),
        m.group("rel"),
        (m.group("spatial") or "").strip(),
        m.group("ref"),
    )


def _parse_add_on_ref(prompt: str):
    # add a butterfly on the left flower
    p = (prompt or "").lower().strip()
    p = re.sub(r"^add\s+(a|an|the)\s+", "add ", p)

    m = re.match(
        r"^add\s+(?P<tgt>\w+)\s+"
        r"(?P<rel>on|onto|in|inside|into|at|near|next to)\s+"
        r"(?:(?:the)\s+)?"
        r"(?:(?P<spatial>left|right|top|bottom)\s+)?"
        r"(?P<ref>\w+)\s*$",
        p,
    )
    if not m:
        return None
    return (
        m.group("tgt"),
        m.group("rel"),
        (m.group("spatial") or "").strip(),
        m.group("ref"),
    )


def main(args: argparse.Namespace):
    setup_tmpdir()

    device = torch.device("cuda") if (torch.cuda.is_available() and not args.cpu) else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    out_dir = args.out_dir or f"/root/SGGE_DM/output/graph_delta_instruct_edits/{args.image_id}"
    os.makedirs(out_dir, exist_ok=True)

    graph_path = os.path.join(args.graph_root, f"{args.image_id}_scene_graph.json")
    graph = load_scene_graph(graph_path)

    img_path = find_image_path_for_graph(graph, graph_path, args.image_id, search_root=args.image_search_root)
    if not img_path:
        raise FileNotFoundError(f"image not found for image_id={args.image_id}")

    image_pil = Image.open(img_path).convert("RGB")
    W, H = image_pil.size

    # ---- intent with ADD override ----
    prompt_l = (args.prompt_tgt or "").lower().strip()
    forced_add = None
    if re.match(r"^add\b", prompt_l):
        forced_add = _parse_add_relation(args.prompt_tgt)
        if forced_add is None:
            forced_add = _parse_add_on_ref(args.prompt_tgt)

    if forced_add is not None:
        tgt_name, rel, ref_spatial, ref_name = forced_add
        intent = "add"
        info = {"tgt": tgt_name, "relation": rel, "ref": ref_name, "ref_spatial": ref_spatial}
    else:
        intent, info = parse_intent(args.prompt_tgt)
        info = dict(info or {})

    _print(f"[INTENT] {intent} | {info}")

    # 给 pipeline 用上下文
    args._device = device
    args._dtype = dtype
    args._intent_info = info
    args._graph_path = graph_path

    out_img = image_pil
    out_path = os.path.join(out_dir, "noedit.png")

    # -------------------------
    # dispatch
    # -------------------------
    if intent in ["attribute", "remove", "replace"]:
        class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]

        if intent == "remove":
            obj_name = str(info.get("obj", "")).lower().strip()
            if not obj_name:
                obj_name = _pick_class_from_prompt(args.prompt_tgt, class_vocab) or "object"

            spatial = _get_spatial_from_info(info, ["obj_spatial", "spatial", "tgt_spatial", "src_spatial"]) or ""
            if not spatial:
                spatial = _extract_spatial_from_prompt(args.prompt_tgt)

            obj_id = require_object_id(
                graph,
                _target_spec(obj_name, spatial),
                device=device,
                clip_local_path=args.clip_local_path,
                err_tag="remove",
            )
            _debug_pick(graph, obj_id, "remove")

            mask_np = load_mask_any(
                graph,
                int(obj_id),
                args.mask_mode,
                graph_path=graph_path,
                yolo_sam_root=args.yolo_sam_root,
            )

            out_img = run_remove(
                args=args,
                image_pil=image_pil,
                mask_np=mask_np,
                obj_name=obj_name,
                device=device,
                dtype=dtype,
            )
            out_path = os.path.join(out_dir, f"remove_{obj_id}_{obj_name}.png")

        elif intent == "replace":
            src_name = str(info.get("src", "object")).lower().strip()
            tgt_name = str(info.get("tgt", "object")).lower().strip()

            spatial = _get_spatial_from_info(info, ["src_spatial", "spatial", "obj_spatial"]) or ""
            if not spatial:
                spatial = _extract_spatial_from_prompt(args.prompt_tgt)

            src_id = require_object_id(
                graph,
                _target_spec(src_name, spatial),
                device=device,
                clip_local_path=args.clip_local_path,
                err_tag="replace-src",
            )
            _debug_pick(graph, src_id, "replace-src")

            src_mask_np = load_mask_any(
                graph,
                int(src_id),
                args.mask_mode,
                graph_path=graph_path,
                yolo_sam_root=args.yolo_sam_root,
            )

            tgt_mask_np = None
            tgt_id = select_object_id(
                graph,
                _target_spec(tgt_name, ""),
                device=device,
                clip_local_path=args.clip_local_path,
                strict=True,
            )
            if tgt_id is not None:
                _debug_pick(graph, tgt_id, "replace-tgt-exists")
                tgt_mask_np = load_mask_any(
                    graph,
                    int(tgt_id),
                    args.mask_mode,
                    graph_path=graph_path,
                    yolo_sam_root=args.yolo_sam_root,
                )
            else:
                _print(f"[REPLACE] tgt '{tgt_name}' not found in graph classes -> SD will draw it")

            out_img = run_replace(
                args=args,
                image_pil=image_pil,
                mask_np=src_mask_np,
                src_name=src_name,
                tgt_name=tgt_name,
                device=device,
                dtype=dtype,
                tgt_mask_np=tgt_mask_np,
                source_image_pil=image_pil,
            )
            out_path = os.path.join(out_dir, f"replace_{src_id}_{src_name}_to_{tgt_name}.png")

        else:  # attribute
            new_val = str(info.get("new", "")).lower().strip()
            tgt_cls = str(info.get("obj", "")).lower().strip()
            if not tgt_cls:
                tgt_cls = _pick_class_from_prompt(args.prompt_tgt, class_vocab) or "object"

            spatial = _get_spatial_from_info(info, ["obj_spatial", "tgt_spatial", "spatial"]) or ""
            if not spatial:
                spatial = _extract_spatial_from_prompt(args.prompt_tgt)

            obj_id = require_object_id(
                graph,
                _target_spec(tgt_cls, spatial),
                device=device,
                clip_local_path=args.clip_local_path,
                err_tag="attribute",
            )
            _debug_pick(graph, obj_id, "attribute")

            mask_np = load_mask_any(
                graph,
                int(obj_id),
                args.mask_mode,
                graph_path=graph_path,
                yolo_sam_root=args.yolo_sam_root,
            )

            tgt_cls_real = _nodes_map(graph).get(int(obj_id), {}).get("class", tgt_cls)

            out_img = run_attribute(
                image_pil=image_pil,
                mask_np=mask_np,
                tgt_cls=tgt_cls_real,
                new_val=new_val,
                prompt_tgt=args.prompt_tgt,
                attr_route=getattr(args, "attr_route", "auto"),
                sd_inpaint_model=args.sd_inpaint_model,
                device=device,
                dtype=dtype,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
            )
            out_path = os.path.join(out_dir, f"attribute_{obj_id}_{tgt_cls_real}_{new_val}.png")

    elif intent == "relation":
        subj_word = str(info.get("subj", "object")).lower().strip()
        obj_word = str(info.get("obj", "object")).lower().strip()
        rel = str(info.get("relation", "on")).lower().strip()

        class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]
        if not subj_word or subj_word in {"to", "of", "the", "a", "an"}:
            subj_word = _pick_class_from_prompt(args.prompt_tgt, class_vocab) or "object"
        if not obj_word or obj_word in {"to", "of", "the", "a", "an"}:
            obj_word = _pick_class_from_prompt(args.prompt_tgt, class_vocab) or "object"

        subj_spatial = _get_spatial_from_info(info, ["subj_spatial", "subj_pos", "subj_location", "spatial_subj"]) or ""
        obj_spatial = _get_spatial_from_info(info, ["obj_spatial", "obj_pos", "obj_location", "spatial_obj"]) or ""
        if not obj_spatial:
            obj_spatial = _extract_spatial_from_prompt(args.prompt_tgt)

        subj_id = require_object_id(
            graph,
            _target_spec(subj_word, subj_spatial),
            device=device,
            clip_local_path=args.clip_local_path,
            err_tag="relation-subj",
        )
        obj_id2 = require_object_id(
            graph,
            _target_spec(obj_word, obj_spatial),
            device=device,
            clip_local_path=args.clip_local_path,
            err_tag="relation-obj",
        )

        _debug_pick(graph, subj_id, "relation-subj")
        _debug_pick(graph, obj_id2, "relation-obj")

        mask_subj = load_mask_any(
            graph,
            int(subj_id),
            args.mask_mode,
            graph_path=graph_path,
            yolo_sam_root=args.yolo_sam_root,
        )
        mask_obj = load_mask_any(
            graph,
            int(obj_id2),
            args.mask_mode,
            graph_path=graph_path,
            yolo_sam_root=args.yolo_sam_root,
        )

        out_img = run_relation(
            args=args,
            image_pil=image_pil,
            mask_np_subj=mask_subj,
            mask_np_obj=mask_obj,
            rel=rel,
            device=device,
            dtype=dtype,
        )
        out_path = os.path.join(out_dir, f"relation_{subj_id}_to_{rel}_{obj_id2}.png")

    elif intent == "add":
        target_center = None

        tgt_name = str(info.get("tgt", "") or info.get("obj", "")).lower().strip()
        rel = str(info.get("relation", "")).lower().strip()
        ref_name = str(info.get("ref", "")).lower().strip()
        ref_spatial = str(info.get("ref_spatial", "")).lower().strip()

        if not tgt_name:
            class_vocab = [n.get("class", "") for n in graph.get("nodes", [])]
            tgt_name = _pick_class_from_prompt(args.prompt_tgt, class_vocab) or "object"

        args._intent_info = getattr(args, "_intent_info", {}) or {}
        args._intent_info["tgt"] = tgt_name

        tgt_name_l = (tgt_name or "").lower().strip()

        # 若有 ref + relation，计算 target_center（支持 on/near/next to）
        if ref_name and rel:
            ref_id = require_object_id(
                graph,
                _target_spec(ref_name, ref_spatial),
                device=device,
                clip_local_path=args.clip_local_path,
                err_tag="add-ref",
            )
            _debug_pick(graph, ref_id, "add-ref")

            ref_mask = load_mask_any(
                graph,
                int(ref_id),
                args.mask_mode,
                graph_path=graph_path,
                yolo_sam_root=args.yolo_sam_root,
            )
            ref_bbox = get_bbox_from_mask(ref_mask, expand=2)

            src_bbox = ref_bbox
            tgt_id_guess = select_object_id(
                graph,
                _target_spec(tgt_name_l, ""),
                device=device,
                clip_local_path=args.clip_local_path,
                strict=True,
            )
            if tgt_id_guess is not None:
                _debug_pick(graph, tgt_id_guess, "add-tgt-size")
                tgt_mask_guess = load_mask_any(
                    graph,
                    int(tgt_id_guess),
                    args.mask_mode,
                    graph_path=graph_path,
                    yolo_sam_root=args.yolo_sam_root,
                )
                try:
                    src_bbox = get_bbox_from_mask(tgt_mask_guess, expand=2)
                except Exception:
                    src_bbox = ref_bbox

            if rel in {"on", "onto", "in", "inside", "into", "at"}:
                x0, y0, x1, y1 = map(float, ref_bbox[:4])
                target_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            elif rel in {"near", "next to"}:
                rel2 = "right of"
                if ref_spatial == "left":
                    rel2 = "left of"
                elif ref_spatial == "right":
                    rel2 = "right of"
                elif ref_spatial == "top":
                    rel2 = "above"
                elif ref_spatial == "bottom":
                    rel2 = "below"
                target_center = compute_target_center_for_relation(rel2, src_bbox, ref_bbox, W, H)
            else:
                target_center = compute_target_center_for_relation(rel, src_bbox, ref_bbox, W, H)

            _print(f"[ADD] target_center=({target_center[0]}, {target_center[1]}) rel={rel} ref={ref_name}({ref_spatial})")

        # 如果图里已有同类物体，给 run_add 做参考
        tgt_mask_np = None
        tgt_id = select_object_id(
            graph,
            _target_spec(tgt_name_l, ""),
            device=device,
            clip_local_path=args.clip_local_path,
            strict=True,
        )
        if tgt_id is not None:
            _debug_pick(graph, tgt_id, "add-exists")
            tgt_mask_np = load_mask_any(
                graph,
                int(tgt_id),
                args.mask_mode,
                graph_path=graph_path,
                yolo_sam_root=args.yolo_sam_root,
            )

        out_img = run_add(
            args=args,
            image_pil=image_pil,
            ref_mask_np=tgt_mask_np,
            source_image_pil=image_pil,
            target_center=target_center,
        )

        safe_tgt = re.sub(r"[^a-z0-9_]+", "_", (tgt_name_l or "obj"))
        safe_ref = re.sub(r"[^a-z0-9_]+", "_", (ref_name or "").lower())
        safe_rel = re.sub(r"[^a-z0-9_]+", "_", (rel or "").lower())
        seed = int(getattr(args, "seed", 0))

        suffix = f"{safe_tgt}"
        if safe_rel and safe_ref:
            suffix += f"_to_{safe_rel}_{(ref_spatial or '')}{safe_ref}"
        suffix += f"_seed{seed}"
        out_path = os.path.join(out_dir, f"add_{suffix}.png")

    # save
    if out_img.size != (W, H):
        out_img = out_img.resize((W, H), Image.BICUBIC)
    out_img.save(out_path)
    _print(f"[OK] saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--sd_inpaint_model", type=str, default="/root/models/sdxl-inpaint")
    p.add_argument("--sd_inpaint_fallback_model", type=str, default=None)

    # selector (used by require_object_id/select_object_id)
    p.add_argument(
        "--clip_local_path",
        type=str,
        default="/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    )

    # io
    p.add_argument("--yolo_sam_root", type=str, default="scripts/yolo_sam2_outputs")
    p.add_argument("--graph_root", type=str, default="./test/scene_graphs_vg")
    p.add_argument("--image_search_root", type=str, default=None)

    # task
    p.add_argument("--image_id", type=str, required=True)
    p.add_argument("--prompt_tgt", type=str, required=True)

    # runtime
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")

    # generic diffusion controls
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, artifacts, unrealistic")

    # ---- legacy remove knobs ----
    p.add_argument("--remove_steps", type=int, default=30)
    p.add_argument("--remove_guidance_scale", type=float, default=3.0)
    p.add_argument("--remove_strength", type=float, default=0.75)

    # graph/mask
    p.add_argument("--mask_mode", type=str, default="mask", choices=["bbox", "mask", "none"])
    p.add_argument("--attr_route", type=str, default="auto", choices=["auto", "recolor", "inpaint"])

    # remove mask shaping
    p.add_argument("--inpaint_dilate", type=int, default=40)
    p.add_argument("--inpaint_blur", type=int, default=18)
    p.add_argument("--mask_erode", type=int, default=0)

    # mask repair knobs
    p.add_argument("--close_k", type=int, default=7)
    p.add_argument("--close_iter", type=int, default=1)
    p.add_argument("--min_area", type=int, default=40)

    # ---- remove pipeline: destroy stage ----
    p.add_argument("--destroy_noise_strength", type=float, default=1.0)
    p.add_argument("--destroy_steps", type=int, default=40)
    p.add_argument("--destroy_guidance_scale", type=float, default=2.0)
    p.add_argument("--destroy_strength", type=float, default=0.9)

    # ---- remove pipeline: CV base fill ----
    p.add_argument("--cv_inpaint_radius", type=int, default=9)
    p.add_argument("--cv_inpaint_method", type=str, default="telea", choices=["telea", "ns"])
    p.add_argument("--cv_dilate", type=int, default=None)
    p.add_argument("--cv_erode", type=int, default=0)

    # ---- remove pipeline: full-hole SD texture fill ----
    p.add_argument("--enable_fill", action="store_true", default=True)
    p.add_argument("--disable_fill", dest="enable_fill", action="store_false")
    p.add_argument("--fill_steps", type=int, default=22)
    p.add_argument("--fill_guidance_scale", type=float, default=1.8)
    p.add_argument("--fill_strength", type=float, default=0.22)
    p.add_argument("--fill_blur", type=int, default=10)

    # ---- remove pipeline: seam repair on ring ----
    p.add_argument("--ring_width", type=int, default=32)
    p.add_argument("--repair_steps", type=int, default=28)
    p.add_argument("--repair_guidance_scale", type=float, default=2.4)
    p.add_argument("--repair_strength", type=float, default=0.22)

    p.add_argument("--enable_cleanup", action="store_true", default=True)
    p.add_argument("--disable_cleanup", dest="enable_cleanup", action="store_false")

    p.add_argument("--remove_trials", type=int, default=3)
    p.add_argument("--auto_tune_remove", action="store_true", default=True)
    p.add_argument("--disable_auto_tune_remove", dest="auto_tune_remove", action="store_false")

    args = p.parse_args()
    main(args)
