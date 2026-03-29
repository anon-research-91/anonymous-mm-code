# scripts/graph_delta/selector.py
import re
from typing import List, Tuple, Dict, Optional

import torch

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    CLIPTokenizer = None
    CLIPTextModel = None
    _TRANSFORMERS_AVAILABLE = False

COLOR_WORDS = [
    "red", "blue", "green", "brown", "black", "white", "yellow", "pink",
    "purple", "gray", "grey", "orange", "beige", "gold", "silver"
]
PATTERN_WORDS = ["striped", "patterned", "plain", "dotted", "checked"]


class SimpleCLIPTextEncoder:
    def __init__(self, device="cuda", clip_local_path: Optional[str] = None):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        model_source = clip_local_path if clip_local_path else "openai/clip-vit-base-patch32"
        local_only = bool(clip_local_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_source, local_files_only=local_only)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = CLIPTextModel.from_pretrained(
            model_source,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=local_only,
        ).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        toks = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
        out = self.model(**toks).last_hidden_state
        emb = out.mean(dim=1)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0)


def auto_select_obj_id_by_clip(graph: dict, target_text: str, device="cuda", clip_local_path: Optional[str] = None) -> int:
    nodes_all = graph.get("nodes", [])
    if len(nodes_all) == 0:
        raise ValueError("graph contains no nodes")
    tt = (target_text or "").lower().strip()
    classes_in_graph = [n.get("class", "").lower() for n in nodes_all]

    if tt in classes_in_graph:
        cand = [n for n in nodes_all if n.get("class", "").lower() == tt]
        if len(cand) == 1:
            return int(cand[0]["id"])

        def bbox_area(n):
            b = n.get("bbox", None)
            if not b or len(b) < 4:
                return -1
            return max(0, (float(b[2]) - float(b[0]))) * max(0, (float(b[3]) - float(b[1])))

        cand = sorted(cand, key=bbox_area, reverse=True)
        return int(cand[0]["id"])

    has_clip_emb = any(n.get("clip_embed", None) is not None for n in nodes_all)
    if not has_clip_emb or not _TRANSFORMERS_AVAILABLE:
        return int(nodes_all[0]["id"])

    enc = SimpleCLIPTextEncoder(device=device, clip_local_path=clip_local_path)
    t_emb = enc.encode(target_text)
    best_sim, best_id = -1e9, None
    for n in nodes_all:
        nid = int(n["id"])
        c = n.get("clip_embed", None)
        if c is None:
            continue
        c_emb = torch.tensor(c, device=enc.device, dtype=t_emb.dtype)
        c_emb = c_emb / c_emb.norm()
        sim = torch.dot(t_emb, c_emb).item()
        if sim > best_sim:
            best_sim, best_id = sim, nid
    if best_id is None:
        raise ValueError("auto obj_id failed")
    return best_id


def _node_get_attrs(node: dict) -> dict:
    for k in ["attributes", "attrs", "attr"]:
        v = node.get(k, None)
        if isinstance(v, dict):
            return v
    return {}


def parse_spatial(prompt: str) -> str:
    p = (prompt or "").lower()
    combos = [
        ("top left", "top-left"),
        ("left top", "top-left"),
        ("upper left", "top-left"),
        ("top right", "top-right"),
        ("right top", "top-right"),
        ("upper right", "top-right"),
        ("bottom left", "bottom-left"),
        ("left bottom", "bottom-left"),
        ("lower left", "bottom-left"),
        ("bottom right", "bottom-right"),
        ("right bottom", "bottom-right"),
        ("lower right", "bottom-right"),
    ]
    for k, v in combos:
        if k in p:
            return v
    if re.search(r"\b(leftmost|far left|on the left|left)\b", p):
        return "left"
    if re.search(r"\b(rightmost|far right|on the right|right)\b", p):
        return "right"
    if re.search(r"\b(middle|center|centre)\b", p):
        return "center"
    if re.search(r"\b(top|upper|at the top)\b", p):
        return "top"
    if re.search(r"\b(bottom|lower|at the bottom|under)\b", p):
        return "bottom"
    return ""


def parse_appearance(prompt: str) -> Dict[str, str]:
    p = (prompt or "").lower()
    out: Dict[str, str] = {}
    for c in COLOR_WORDS:
        if re.search(rf"\b{re.escape(c)}\b", p):
            out["color"] = c
            break
    for pat in PATTERN_WORDS:
        if re.search(rf"\b{re.escape(pat)}\b", p):
            out["pattern"] = pat
            break
    return out


def pick_target_noun(prompt: str, class_vocab: List[str]) -> str:
    p = (prompt or "").lower()
    vocab_sorted = sorted(set([c.lower() for c in class_vocab if c]), key=len, reverse=True)
    for c in vocab_sorted:
        if re.search(rf"\b{re.escape(c)}\b", p):
            return c
    return ""


def parse_target_object(prompt: str, class_vocab: List[str]) -> Dict:
    cls = pick_target_noun(prompt, class_vocab)
    return {
        "class": (cls or "").lower().strip(),
        "spatial": parse_spatial(prompt),
        "appearance": parse_appearance(prompt),
    }


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


def select_object_id(graph: dict, target: Dict, *, device="cuda", clip_local_path: Optional[str] = None) -> int:
    nodes_all = graph.get("nodes", [])
    if not nodes_all:
        raise ValueError("graph contains no nodes")

    tgt_cls = (target.get("class") or "").lower().strip()
    spatial = (target.get("spatial") or "").lower().strip()
    appearance: Dict[str, str] = target.get("appearance") or {}

    cand = nodes_all
    if tgt_cls:
        cand = [n for n in nodes_all if (n.get("class", "").lower() == tgt_cls)]
        if not cand:
            return auto_select_obj_id_by_clip(graph, tgt_cls, device=device, clip_local_path=clip_local_path)

    def appearance_score(n: dict) -> int:
        if not appearance:
            return 0
        attrs = _node_get_attrs(n)
        if not attrs:
            return 0
        score = 0
        for k, v in appearance.items():
            if str(attrs.get(k, "")).lower() == str(v).lower():
                score += 1
        return score

    if appearance:
        cand = sorted(cand, key=lambda n: (appearance_score(n), _bbox_area(n)), reverse=True)

    if spatial and cand:
        centers = [(n, *_bbox_center(n)) for n in cand]
        if spatial == "left":
            cand = [min(centers, key=lambda t: t[1])[0]]
        elif spatial == "right":
            cand = [max(centers, key=lambda t: t[1])[0]]
        elif spatial == "top":
            cand = [min(centers, key=lambda t: t[2])[0]]
        elif spatial == "bottom":
            cand = [max(centers, key=lambda t: t[2])[0]]
        elif spatial == "center":
            mx = sum(t[1] for t in centers) / len(centers)
            my = sum(t[2] for t in centers) / len(centers)
            cand = [min(centers, key=lambda t: (t[1] - mx) ** 2 + (t[2] - my) ** 2)[0]]
        elif "-" in spatial:
            vpart, hpart = spatial.split("-", 1)
            tmp = centers
            tmp = sorted(tmp, key=lambda t: t[2]) if vpart == "top" else sorted(tmp, key=lambda t: -t[2])
            tmp = sorted(tmp, key=lambda t: t[1]) if hpart == "left" else sorted(tmp, key=lambda t: -t[1])
            cand = [tmp[0][0]]

    if len(cand) > 1:
        cand = [max(cand, key=_bbox_area)]
    return int(cand[0]["id"])
