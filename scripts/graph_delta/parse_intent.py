# scripts/graph_delta/parse_intent.py
import re
from typing import Tuple, Dict

COLOR_WORDS = [
    "red", "blue", "green", "brown", "black", "white", "yellow", "pink",
    "purple", "gray", "grey", "orange", "beige", "gold", "silver"
]

REPLACE_PATTERNS = [
    r"\breplace\b\s+(a |an |the )?(.+?)\s+\bwith\b\s+(a |an |the )?(.+)$",
    r"\bturn\b\s+(a |an |the )?(.+?)\s+\binto\b\s+(a |an |the )?(.+)$",
]
REMOVE_PATTERNS = [
    r"\b(remove|delete|erase)\b\s+(a |an |the )?((left|right|top|bottom|upper|lower)\s+)?(\w+)",
    r"\b(without|no)\b\s+(a |an |the )?((left|right|top|bottom|upper|lower)\s+)?(\w+)",
]
RELATION_PATTERNS = [
    r"(move|put|place)\s+(a |an |the )?(.+?)\s+(left of|right of|on|under|above|below|next to|holding)\s+(a |an |the )?(.+)$",
    r"(.+?)\s+(left of|right of|on|under|above|below|next to|holding)\s+(.+)$",
]
ADD_PATTERNS = [
    r"\badd\b\s+(a |an |the )?(.+)$",
    r"\b(place|put|insert)\b\s+(a |an |the )?(.+)$",
]

SPATIAL_WORDS = ["left", "right", "top", "bottom", "upper", "lower", "center", "middle"]


def parse_noun_phrase(text: str) -> Dict[str, str]:
    t = (text or "").lower().strip()
    t = re.sub(r"\b(a|an|the)\b", "", t).strip()
    toks = [x for x in re.split(r"\s+", t) if x]
    spatial = []
    noun = ""
    for w in toks:
        if w in ["upper"]:
            w = "top"
        if w in ["lower"]:
            w = "bottom"
        if w in ["middle"]:
            w = "center"
        if w in SPATIAL_WORDS:
            spatial.append(w)
        else:
            noun = w
    sp = ""
    if spatial:
        if "top" in spatial and "left" in spatial:
            sp = "top-left"
        elif "top" in spatial and "right" in spatial:
            sp = "top-right"
        elif "bottom" in spatial and "left" in spatial:
            sp = "bottom-left"
        elif "bottom" in spatial and "right" in spatial:
            sp = "bottom-right"
        elif "left" in spatial:
            sp = "left"
        elif "right" in spatial:
            sp = "right"
        elif "top" in spatial:
            sp = "top"
        elif "bottom" in spatial:
            sp = "bottom"
        elif "center" in spatial:
            sp = "center"
    return {"noun": noun, "spatial": sp}


def _parse_add_on_ref(prompt: str):
    """
    add a butterfly on the left flower
    add butterfly near right plate
    add spoon next to knife
    """
    p = (prompt or "").lower().strip()
    p = re.sub(r"^add\s+(a|an|the)\s+", "add ", p)

    m = re.match(
        r"^add\s+(?P<tgt>\w+)\s+"
        r"(?P<rel>on|onto|in|inside|into|at|near|next to)\s+"
        r"(?:(?:the)\s+)?"
        r"(?:(?P<spatial>left|right|top|bottom|upper|lower|center|middle)\s+)?"
        r"(?P<ref>\w+)\s*$",
        p,
    )
    if not m:
        return None

    sp = (m.group("spatial") or "").strip()
    if sp == "upper":
        sp = "top"
    if sp == "lower":
        sp = "bottom"
    if sp == "middle":
        sp = "center"

    return (m.group("tgt"), m.group("rel"), sp, m.group("ref"))


def parse_intent(prompt: str) -> Tuple[str, Dict]:
    p = (prompt or "").lower().strip()

    # ---- STRONG ADD PRIORITY ----
    # 只要是 add/insert/put/place 开头，优先按 add 解析，避免落入 RELATION_PATTERNS 的“X on Y”
    if re.match(r"^(add|insert|put|place)\b", p):
        # 优先解析 add X on/near/next to Y
        if p.startswith("add "):
            r = _parse_add_on_ref(p)
            if r is not None:
                tgt, rel, ref_spatial, ref = r
                return "add", {"tgt": tgt, "relation": rel, "ref": ref, "ref_spatial": ref_spatial}

        # fallback: 原 add pattern，只取 tgt noun
        for pat in ADD_PATTERNS:
            m = re.search(pat, p)
            if m:
                tgt_phrase = m.group(2)
                tgt_np = parse_noun_phrase(tgt_phrase)
                return "add", {"tgt": tgt_np["noun"]}

    # ---- special move-to-center ----
    m_center = re.search(
        r"(move|put|place)\s+(a |an |the )?(\w+)\s+to the center of (the )?(image|scene|photo|picture)",
        p,
    )
    # strong rule: move X to the left/right/above/below of (the) [spatial] Y
    m_move = re.search(
        r"^(move|put|place)\s+(?:a |an |the )?(\w+)\s+to\s+the\s+"
        r"(left of|right of|above|below|under)\s+(?:the )?"
        r"(left|right|top|bottom)?\s*(\w+)\s*$",
        p,
    )
    if m_move:
        subj = m_move.group(2)
        rel = m_move.group(3)
        obj_spatial = (m_move.group(4) or "").strip()
        obj = m_move.group(5)
        info = {"subj": subj, "relation": rel, "obj": obj}
        if obj_spatial:
            info["obj_spatial"] = obj_spatial
        return "relation", info

    if m_center:
        subj = m_center.group(3)
        return "relation", {"subj": subj, "relation": "center", "obj": subj}

    # ---- relation ----
    for pat in RELATION_PATTERNS:
        m = re.search(pat, p)
        if not m:
            continue
        if len(m.groups()) >= 6 and m.group(1) in ["move", "put", "place"]:
            subj_phrase = m.group(3)
            rel = m.group(4)
            obj_phrase = m.group(6)
            subj_np = parse_noun_phrase(subj_phrase)
            obj_np = parse_noun_phrase(obj_phrase)
            return "relation", {
                "subj": subj_np["noun"],
                "subj_spatial": subj_np["spatial"],
                "relation": rel,
                "obj": obj_np["noun"],
                "obj_spatial": obj_np["spatial"],
            }

        subj_phrase = m.group(1)
        rel = m.group(2)
        obj_phrase = m.group(3)
        subj_np = parse_noun_phrase(subj_phrase)
        obj_np = parse_noun_phrase(obj_phrase)
        return "relation", {
            "subj": subj_np["noun"],
            "subj_spatial": subj_np["spatial"],
            "relation": rel,
            "obj": obj_np["noun"],
            "obj_spatial": obj_np["spatial"],
        }

    # ---- attribute ----
    for c in COLOR_WORDS:
        if re.search(rf"\b{re.escape(c)}\b", p):
            return "attribute", {"attr_name": "color", "new": c}

    # ---- replace ----
    for pat in REPLACE_PATTERNS:
        m = re.search(pat, p)
        if m:
            src_phrase = m.group(2)
            tgt_phrase = m.group(4)
            src_np = parse_noun_phrase(src_phrase)
            tgt_np = parse_noun_phrase(tgt_phrase)
            return "replace", {
                "src": src_np["noun"],
                "src_spatial": src_np["spatial"],
                "tgt": tgt_np["noun"],
            }

    # ---- add (non-leading add, just in case) ----
    for pat in ADD_PATTERNS:
        m = re.search(pat, p)
        if m:
            tgt_phrase = m.group(2)
            tgt_np = parse_noun_phrase(tgt_phrase)
            return "add", {"tgt": tgt_np["noun"]}

    # ---- remove ----
    for pat in REMOVE_PATTERNS:
        m = re.search(pat, p)
        if m:
            noun = m.group(5)
            return "remove", {"obj": noun}

    return "none", {}
