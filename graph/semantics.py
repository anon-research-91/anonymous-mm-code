# graph/semantics.py
# 图谱语义 verbalization + ΔG → 文本嵌入方向向量（无训练）

from typing import List, Dict, Tuple, Optional
import torch

# 为了避免循环引用，这里只在函数内部 import GraphDelta 相关
# from graph.graph_delta import GraphDelta, graph_delta_to_instruction


def _node_display_name(node: Dict) -> str:
    """把一个 node 变成一句简短描述，比如 'red chair' / 'chair'."""
    cls = node.get("class", "object")
    attrs = node.get("attrs", [])
    if isinstance(attrs, str):
        attrs = [attrs]
    attrs = [a for a in attrs if a]
    if attrs:
        return f"{', '.join(attrs)} {cls}"
    else:
        return cls


def graph_to_text(graph: Dict) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    id2cls = {int(n.get("id")): n.get("class", "object") for n in nodes if n.get("id") is not None}

    obj_phrases = [_node_display_name(n) for n in nodes]
    obj_part = ("a scene with " + ", ".join(obj_phrases)) if obj_phrases else "a scene"

    rel_phrases = []
    for e in edges[:10]:
        r = (e.get("predicate") or e.get("relation") or "").strip()
        if not r:
            continue

        # support (subject_id, object_id) OR (subject, object) OR (source, target)
        sid = e.get("subject_id", None)
        oid = e.get("object_id", None)

        if sid is not None and oid is not None:
            s = id2cls.get(int(sid), str(sid))
            o = id2cls.get(int(oid), str(oid))
        else:
            s = e.get("subject") or e.get("source") or ""
            o = e.get("object") or e.get("target") or ""
            if not (s and o):
                continue

        rel_phrases.append(f"{s} {r} {o}")

    rel_part = (" In the scene, " + "; ".join(rel_phrases) + ".") if rel_phrases else ""
    return obj_part + "." + rel_part

    """
    把 scene graph verbalize 成一段普通英文描述。
    不追求特别复杂，只要能提供一个全局语义“背景”即可。
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 1) 对象列表
    obj_phrases = []
    for n in nodes:
        obj_phrases.append(_node_display_name(n))
    if obj_phrases:
        obj_part = "a scene with " + ", ".join(obj_phrases)
    else:
        obj_part = "a scene"

    # 2) 关系列表（选前若干个，避免太长）
    rel_phrases = []
    for e in edges[:10]:
        # 这里假设 edge 里可能存的是 id，也可能是 class/name，做个健壮处理
        s = e.get("subject", "")
        o = e.get("object", "")
        r = e.get("predicate", "")
        if not (s and o and r):
            continue
        rel_phrases.append(f"{s} {r} {o}")

    if rel_phrases:
        rel_part = " In the scene, " + "; ".join(rel_phrases) + "."
    else:
        rel_part = ""

    return obj_part + "." + rel_part


def build_target_text_from_deltas(graph: Dict, deltas: List) -> str:
    """
    不真正修改 graph 结构，而是：原图谱文本 + 若干条 ΔG 的自然语言描述，
    作为“目标语义”文本。
    """
    # 延迟 import，避免循环依赖
    from graph.graph_delta import graph_delta_to_instruction

    base_text = graph_to_text(graph)

    delta_texts = []
    for d in deltas:
        try:
            t = graph_delta_to_instruction(d, graph)
        except Exception:
            t = None
        if t:
            delta_texts.append(t)

    if not delta_texts:
        return base_text

    delta_part = " After editing, " + " And ".join(delta_texts) + "."
    return base_text + delta_part


def compute_semantic_direction_from_graph(
    pipe,
    graph: Dict,
    deltas: List,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 pipeline 自带的 tokenizer/text_encoder：
    - G_text  = graph_to_text(G)
    - Gp_text = build_target_text_from_deltas(G, ΔG)
    得到 e_G, e_G'，方向 d = norm(e_G' - e_G)

    返回:
        e_G, e_Gp, d  形状均为 (1, L, D)，可直接喂给 diffusers.
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_G = graph_to_text(graph)
    text_Gp = build_target_text_from_deltas(graph, deltas)

    with torch.no_grad():
        toks_G = tokenizer(
            [text_G],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        toks_Gp = tokenizer(
            [text_Gp],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        e_G = text_encoder(**toks_G)[0]   # (1, L, D)
        e_Gp = text_encoder(**toks_Gp)[0] # (1, L, D)

    d = e_Gp - e_G
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
    return e_G, e_Gp, d
