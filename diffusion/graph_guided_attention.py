# diffusion/graph_guided_attention.py
from typing import Dict, Optional

import torch
from torch import nn

# ✅ 兼容不同 diffusers 版本：
#   - 新版有 AttnProcessor 类（真正的基类）
#   - 老版可能只有 AttentionProcessor
try:
    from diffusers.models.attention_processor import AttnProcessor as _BaseAttnProcessor
except ImportError:
    # 老版本：AttentionProcessor 本身就是可继承的基类
    from diffusers.models.attention_processor import AttentionProcessor as _BaseAttnProcessor


class GraphGuidedAttnProcessor(_BaseAttnProcessor):
    """
    封装原始 attention processor，在 cross-attn 时把 graph tokens 拼到文本 token 后面：

      K_all = concat(K_text, K_graph)
      V_all = concat(V_text, V_graph)

    这里不自己实现 attention，只是改 encoder_hidden_states，
    然后调用原来的 processor。
    """

    def __init__(
        self,
        base_processor: _BaseAttnProcessor,
        graph_tokens: Optional[torch.Tensor] = None,
        alpha_graph: float = 1.0,
    ):
        super().__init__()
        self.base_processor = base_processor
        self.graph_tokens = graph_tokens
        self.alpha_graph = alpha_graph  # 控制 graph token 的“音量”

    def set_graph_tokens(self, graph_tokens: Optional[torch.Tensor]):
        self.graph_tokens = graph_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        注意：不同 diffusers 版本 attention processor 的 __call__ 签名略有区别，
        所以这里用 **kwargs 把多余参数兜住。
        """
        if encoder_hidden_states is not None and self.graph_tokens is not None:
            g = self.graph_tokens

            # broadcast to batch size & 对齐 device / dtype
            if g.device != encoder_hidden_states.device:
                g = g.to(encoder_hidden_states.device)
            if g.dtype != encoder_hidden_states.dtype:
                g = g.to(encoder_hidden_states.dtype)

            if g.shape[0] != encoder_hidden_states.shape[0]:
                # (1, N, D) -> (B, N, D)
                g = g.expand(encoder_hidden_states.shape[0], -1, -1)

            g = self.alpha_graph * g
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, g], dim=1
            )

        # 调用原始 processor 完成 attention 计算
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


def wrap_unet_with_graph_guidance(
    unet,
    graph_tokens: torch.Tensor,
    alpha_graph: float = 1.0,
) -> Dict[str, _BaseAttnProcessor]:
    """
    给 UNet 的 cross-attention 模块（attn2）套上 GraphGuidedAttnProcessor。
    返回一个 dict，保存原始 processor，方便之后恢复。
    """
    orig_processors: Dict[str, _BaseAttnProcessor] = dict(unet.attn_processors)
    new_processors = {}

    for name, proc in orig_processors.items():
        # attn2 通常是 cross-attention，attn1 是 self-attn
        if "attn2" in name:
            new_processors[name] = GraphGuidedAttnProcessor(
                base_processor=proc,
                graph_tokens=graph_tokens,
                alpha_graph=alpha_graph,
            )
        else:
            new_processors[name] = proc

    unet.set_attn_processor(new_processors)
    return orig_processors


def restore_unet_attn_processors(
    unet,
    orig_processors: Dict[str, _BaseAttnProcessor],
):
    """
    恢复 UNet 原始的 attention processor（去掉图引导）。
    """
    unet.set_attn_processor(orig_processors)
