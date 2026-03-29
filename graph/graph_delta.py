# graph/graph_delta.py
from typing import TypedDict, Literal, Optional, Dict, Any, List


# ---- 1) Graph Delta 数据结构 --------------------------------------------

DeltaType = Literal["attribute", "relation", "add", "remove"]


class GraphDelta(TypedDict, total=False):
    # 通用字段
    type: DeltaType

    # 属性变化: Cup.color: blue -> red
    target_obj: int           # node.id
    attr_name: str            # "color", "style", ...
    old: str
    new: str

    # 关系变化: Person --hold--> Cup
    subject_id: int           # subject node id
    object_id: int            # object node id
    relation: str             # "hold", "on", "left of", ...

    # 增删 node
    object_class: str         # "tree", "cup", ...
    location: str             # "in the background", "on the table", ...
    description: str          # 自然语言描述: "the second cup on the right"


# ---- 2) 小工具: 通过 node.id 找 node ------------------------------------


def _find_node_by_id(graph: Dict[str, Any], nid: int) -> Optional[Dict[str, Any]]:
    for n in graph.get("nodes", []):
        if int(n.get("id", -1)) == int(nid):
            return n
    return None


# ---- 3) Graph Delta -> 文本指令 -----------------------------------------


def graph_delta_to_instruction(delta: GraphDelta,
                               graph: Dict[str, Any]) -> str:
    """
    把单个 GraphDelta 映射成一条自然语言指令（强局部约束版）。
    """

    dtype = delta.get("type", "attribute")

    # === 1) 属性变化 ======================================================
    if dtype == "attribute":
        target_id = delta["target_obj"]
        node = _find_node_by_id(graph, target_id)
        obj_class = node.get("class", "object") if node is not None else "object"

        attr_name = delta.get("attr_name", "attribute")
        old = delta.get("old", None)
        new = delta.get("new", None)

        # ✅ hard local constraints（关键：防止过改/漂）
        keep_clause = (
            f" Keep the {obj_class}'s shape, position, size, and texture unchanged."
            f" Do not modify anything outside the {obj_class} region."
        )

        if old is not None and new is not None:
            return (
                f"Change the {attr_name} of the {obj_class} from {old} to {new}."
                + keep_clause
            )
        elif new is not None:
            return (
                f"Change the {attr_name} of the {obj_class} to {new}."
                + keep_clause
            )
        else:
            return f"Modify the {attr_name} of the {obj_class}." + keep_clause

    # === 2) 关系变化 ======================================================
    if dtype == "relation":
        sid = delta["subject_id"]
        oid = delta["object_id"]
        rel = delta.get("relation", "interact with")

        s_node = _find_node_by_id(graph, sid)
        o_node = _find_node_by_id(graph, oid)

        s_class = s_node.get("class", "object") if s_node is not None else "object"
        o_class = o_node.get("class", "object") if o_node is not None else "object"

        keep_clause = (
            f" Keep both the {s_class} and the {o_class} appearances unchanged."
            f" Do not change the background."
        )

        return f"Make the {s_class} {rel} the {o_class}." + keep_clause

    # === 3) 添加节点 ======================================================
    if dtype == "add":
        obj_cls = delta.get("object_class", "object")
        loc = delta.get("location", None)
        if loc:
            return f"Add a {obj_cls} {loc}."
        else:
            return f"Add a {obj_cls}."

    # === 4) 删除节点 ======================================================
    if dtype == "remove":
        desc = delta.get("description", None)
        if desc:
            return f"Remove {desc}."
        obj_cls = delta.get("object_class", "object")
        loc = delta.get("location", None)
        if loc:
            return f"Remove the {obj_cls} that is {loc}."
        else:
            return f"Remove the {obj_cls}."

    return "Modify the image according to the graph change."


# ---- 4) ✅ 指令 ensemble（轻量等价多样化） ------------------------------


def graph_delta_to_instruction_ensemble(delta: GraphDelta,
                                        graph: Dict[str, Any],
                                        k: int = 3) -> List[str]:
    """
    同一 delta 生成多条等价指令，便于投票/选优。
    不依赖 LLM，先手写变体就够用。
    """
    base = graph_delta_to_instruction(delta, graph)

    variants = [
        base,
        base.replace("Change", "Modify").replace("from", "changing from"),
        base + " Only edit the masked area.",
        base.replace("Do not modify anything outside", "Leave everything outside unchanged, do not modify"),
    ]

    # 去重保持次序
    uniq = []
    for v in variants:
        if v not in uniq:
            uniq.append(v)

    return uniq[:k]


# ---- 5) 多 Delta 合并 ---------------------------------------------------


def combine_deltas_to_instruction(deltas: List[GraphDelta],
                                  graph: Dict[str, Any]) -> str:
    if not deltas:
        return "Modify the image."

    texts = [graph_delta_to_instruction(d, graph) for d in deltas]

    if len(texts) == 1:
        return texts[0]

    first = texts[0]
    rest = texts[1:]
    rest_join = " Then, ".join(rest)
    return f"{first} Then, {rest_join}"
