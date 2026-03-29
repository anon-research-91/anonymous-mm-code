# graph/graph_tokens.py
import json
import os
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


class GraphTokenBuilder(nn.Module):
    """
    把 scene_graph JSON 里的 clip_embed / graph_embed
    映射到 Stable Diffusion 文本 encoder 的 hidden dim 上，
    封装为一串 “伪 token”：
      G = [ g_obj1, g_obj2, ..., g_rel, g_graph ]
    """

    def __init__(
        self,
        clip_dim: int,
        text_dim: int,
        device: str = "cuda",
        use_mlp: bool = True,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.text_dim = text_dim
        self.device = device

        if clip_dim == text_dim and not use_mlp:
            self.proj = nn.Identity()
        else:
            # 简单两层 MLP 做投影（目前不训练，当作固定映射也可以）
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, text_dim),
                nn.LayerNorm(text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim),
            )

        self.to(device)

    def forward(
        self,
        graph: Dict[str, Any],
        selected_obj_ids: Optional[List[int]] = None,
        use_relations: bool = False,
        include_graph_embed: bool = True,
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        graph: 解析后的 scene_graph dict
        selected_obj_ids: 只保留这些节点（对象级控制时用）
        返回:
          graph_tokens: (1, N_tokens, text_dim)
          nodeid2idx:   {node_id -> 在 graph_tokens 里的 token index}
        """
        nodes = graph.get("nodes", [])
        rels = graph.get("edges", [])
        tokens: List[torch.Tensor] = []
        nodeid2idx: Dict[int, int] = {}

        # 1) 对象节点 token
        for node in nodes:
            nid = int(node["id"])
            if selected_obj_ids is not None and nid not in selected_obj_ids:
                continue
            clip_emb = node.get("clip_embed", None)
            if clip_emb is None:
                continue
            emb = torch.tensor(clip_emb, dtype=torch.float32, device=self.device)
            if emb.numel() != self.clip_dim:
                raise ValueError(
                    f"node {nid} clip_embed dim={emb.numel()}, expected {self.clip_dim}"
                )
            proj = self.proj(emb)  # (text_dim,)
            nodeid2idx[nid] = len(tokens)
            tokens.append(proj)

        # 2) 关系 token（可选）
        if use_relations:
            for e in rels:
                clip_rel = e.get("clip_embed", None)
                if clip_rel is None:
                    continue
                emb = torch.tensor(clip_rel, dtype=torch.float32, device=self.device)
                if emb.numel() != self.clip_dim:
                    continue
                proj = self.proj(emb)
                tokens.append(proj)

        # 3) 图级全局 token（可选）
        if include_graph_embed and "graph_embed" in graph:
            g_emb = torch.tensor(
                graph["graph_embed"], dtype=torch.float32, device=self.device
            )
            if g_emb.numel() == self.clip_dim:
                proj = self.proj(g_emb)
                tokens.append(proj)

        if len(tokens) == 0:
            raise ValueError("No graph tokens constructed; check clip_embed fields.")

        graph_tokens = torch.stack(tokens, dim=0).unsqueeze(0)  # (1, N, D)
        return graph_tokens, nodeid2idx


def load_scene_graph(graph_path: str) -> Dict[str, Any]:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(graph_path)
    with open(graph_path, "r") as f:
        graph = json.load(f)
    return graph
