# diffusion/instruct_pix2pix_editor.py

import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

from graph.semantics import compute_semantic_direction_from_graph


@dataclass
class InstructEditConfig:
    num_inference_steps: int = 40
    guidance_scale: float = 5.0
    image_guidance_scale: float = 1.8
    seed: int = 123


class InstructPix2PixEditor:
    """
    轻量封装 StableDiffusionInstructPix2PixPipeline：
    - 支持 mask 局部编辑（在像素域里混合）
    - 支持可选的 graph-guided 语义方向控制
    """

    def __init__(self, model_id: str, device: torch.device, torch_dtype: torch.dtype):
        self.device = device

        try:
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                local_files_only=True,
            ).to(device)
        except OSError as e:
            raise OSError(
                f"Failed to load InstructPix2Pix model from {model_id}.\n"
                f"Error: {str(e)}\n"
                f"Please ensure the model directory contains model_index.json and all required components.\n"
            ) from e

        if device.type == "cuda":
            self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def _run_ip2p_core(
        self,
        image_pil: Image.Image,
        instruction: str,
        cfg: InstructEditConfig,
        graph=None,
        deltas: Optional[List] = None,
        graph_lambda: float = 1.0,
    ) -> Image.Image:
        """
        核心调用 InstructPix2PixPipeline：
        - 若 graph/deltas 提供，则使用基于图谱的 prompt_embeds；
        - 否则使用原始 instruction 作为 prompt。
        """
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        # -------- 图谱语义方向控制（可选） --------
        if graph is not None and deltas is not None and graph_lambda != 0.0:
            e_G, e_Gp, d = compute_semantic_direction_from_graph(
                self.pipe, graph, deltas, self.device
            )
            guided_embeds = e_Gp + graph_lambda * d  # (1, L, D)

            # 负提示词，简单用一条通用的低质量抑制，可以按需改
            neg_prompt = "blurry, low quality, artifacts, distorted, deformed"
            toks_neg = self.pipe.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            neg_embeds = self.pipe.text_encoder(**toks_neg)[0]

            out = self.pipe(
                image=image_pil,
                prompt=instruction,
                negative_prompt="text, logo, watermark, frame, border, duplicate, extra object, blurry, low quality, artifacts",
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                image_guidance_scale=cfg.image_guidance_scale,
                generator=gen,
            )
        else:
            # -------- 纯文本 IP2P 路线（兼容旧版本） --------
            out = self.pipe(
                image=image_pil,
                prompt=instruction,
                negative_prompt="text, logo, watermark, frame, border, duplicate, extra object, blurry, low quality, artifacts",
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                image_guidance_scale=cfg.image_guidance_scale,
                generator=gen,
            )

        return out.images[0]

    @torch.inference_mode()
    def run_edit(
        self,
        image_pil: Image.Image,
        instruction: str,
        cfg: InstructEditConfig,
        mask: Optional[torch.Tensor] = None,
        graph=None,
        deltas: Optional[List] = None,
        graph_lambda: float = 1.0,
    ) -> Image.Image:
        """
        统一入口：
        - image_pil: 原始图像 (PIL)
        - instruction: 文本指令（仍然保留，用于 baseline 和日志）
        - cfg: 推理配置
        - mask: (H, W) 0/1 tensor，可选；若提供则只在该区域应用编辑
        - graph / deltas / graph_lambda: 图谱语义控制（可选）

        返回：编辑后的 PIL Image，尺寸与输入一致。
        """
        W, H = image_pil.size

        edited_full = self._run_ip2p_core(
            image_pil=image_pil,
            instruction=instruction,
            cfg=cfg,
            graph=graph,
            deltas=deltas,
            graph_lambda=graph_lambda,
        )

        if mask is None:
            # 无 mask，整图编辑
            return edited_full.resize((W, H), resample=Image.BICUBIC)

        # -------- 使用 mask 做局部混合（像素域） --------
        # mask: [H, W] 0/1
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().float().numpy()
        else:
            mask_np = np.array(mask).astype(np.float32)
            if mask_np.max() > 1.0:
                mask_np = mask_np / 255.0

        mask_np = np.clip(mask_np, 0.0, 1.0)

        # resize mask 到与图像一致
        if mask_np.shape != (H, W):
            from PIL import Image as _Image
            m_img = _Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (W, H), resample=_Image.NEAREST
            )
            mask_np = np.array(m_img).astype(np.float32) / 255.0

        mask_np = mask_np[..., None]  # [H,W,1] 便于广播

        orig_np = np.array(image_pil).astype(np.float32)
        edit_np = np.array(edited_full.resize((W, H), resample=Image.BICUBIC)).astype(
            np.float32
        )

        out_np = mask_np * edit_np + (1.0 - mask_np) * orig_np
        out_np = np.clip(out_np, 0, 255).astype(np.uint8)

        return Image.fromarray(out_np)
