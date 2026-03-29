# diffusion/blended_diffusion.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List

from diffusers import StableDiffusionPipeline


@dataclass
class BlendedEditConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    # 噪声种子，为了复现结果可控
    seed: int = 42
    # 是否用 source prompt 参与背景部分（先简单用 False）
    use_source_prompt_for_bg: bool = False


class BlendedDiffusionEditor:
    def __init__(
        self,
        sd_model: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        """
        sd_model: Stable Diffusion 模型名字或本地路径
                  例如: "runwayml/stable-diffusion-v1-5" 或 "/path/to/sd"
        """
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            dtype=torch_dtype,
        ).to(device)
        # 如果显存紧张：
        # self.pipe.enable_xformers_memory_efficient_attention()

    @torch.no_grad()
    def encode_image(self, image_pil):
        pipe = self.pipe

        # 1. 用 VaeImageProcessor 做预处理（返回已经是 torch.Tensor）
        #   - 不要再写 return_tensors="pt"
        #   - 不要把 image_processor 当函数调用
        image = pipe.image_processor.preprocess(image_pil)  # (B, C, H, W)

        # 2. 放到正确的 device 和 dtype（和 UNet 保持一致）
        image = image.to(device=pipe.device, dtype=pipe.unet.dtype)

        # 3. 送进 VAE 做编码，得到 latent
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor  # 一般是 0.18215

        return latents


    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        用 pipeline 内部方法获取 text encoder hidden states
        """
        pipe = self.pipe
        # 这里用 pipeline 的私有方法 _encode_prompt，简单直接
        text_embeds = pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        # 返回 shape: (2, 77, 768)  -> [uncond, cond] 拼在一起
        return text_embeds

    @torch.no_grad()
    def run_blended_edit(
        self,
        image_pil,
        mask_tensor: torch.Tensor,
        prompt_src: str,
        prompt_tgt: str,
        cfg: Optional[BlendedEditConfig] = None,
    ):
        """
        image_pil: 原始 PIL Image
        mask_tensor: torch.Tensor, shape (H, W)，值为 0/1，1 表示编辑区域（来自 SAM2 npy）
        prompt_src: 原图语义描述（可选）
        prompt_tgt: 目标编辑文本
        """
        if cfg is None:
            cfg = BlendedEditConfig()

        pipe = self.pipe
        device = self.device
        guidance_scale = cfg.guidance_scale
        num_inference_steps = cfg.num_inference_steps

        # 用全局随机种子控制噪声即可，避免跟老版本 API 打架
        torch.manual_seed(cfg.seed)

        # === 1. 编码图像到 latent: z_0 ===
        z_0 = self.encode_image(image_pil)  # (1,4,h,w)
        b, c, h, w = z_0.shape

        # === 2. 文本编码 ===
        # text_embeds: (2, 77, 768): [uncond; target_prompt]
        text_embeds = self.encode_prompt(prompt_tgt)

        # 如果你想用 source prompt 约束背景，可以在这里也 encode 一份
        # text_embeds_src = self.encode_prompt(prompt_src)

        # === 3. 准备 scheduler & 时间步 ===
        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)

        # 采样初始噪声，并把 z_0 推到最开始的时间步 t_0
        noise = torch.randn_like(z_0)
        # 第一个时间步（通常是最大 t）
        t_start = scheduler.timesteps[0]
        z_t_init = scheduler.add_noise(z_0, noise, t_start)

        # 用于编辑路径的 latent
        latents_edit = z_t_init.clone()

        # === 4. 处理 mask: resize 到 latent 空间 ===
        # mask_tensor 是 (H_img, W_img) 0/1
        mask = mask_tensor.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        # 先在 CPU 上插值到 latent 大小 (h, w)
        mask_latent = F.interpolate(mask, size=(h, w), mode="nearest")
        # 确保是 0/1
        mask_latent = (mask_latent > 0.5).float()  # (1,1,h,w)
        # ⚠️ 关键：把 mask 搬到跟 latent 一样的 device 和 dtype
        mask_latent = mask_latent.to(device=device, dtype=z_0.dtype)

        # === 5. 反向扩散循环（Blended Diffusion） ===
        for i, t in enumerate(scheduler.timesteps):
            # 5.1 计算当前 t 对应的“原图加噪 latent”：z_t_orig
            z_t_orig = scheduler.add_noise(z_0, noise, t)  # (1,4,h,w)

            # 5.2 对编辑路径走一步扩散
            # classifier-free guidance:
            latent_model_input = torch.cat([latents_edit] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds,
            ).sample  # (2,4,h,w)

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents_edit = scheduler.step(
                noise_pred_cfg, t, latents_edit
            ).prev_sample  # (1,4,h,w)

            # 5.3 Blended Diffusion：掩膜内用编辑结果，掩膜外用原图路径
            # 你也可以做一个时间相关的 alpha_t，这里先用固定 1.0
            alpha_t = 1.0
            blended_mask = mask_latent * alpha_t  # (1,1,h,w)

            latents_edit = blended_mask * latents_edit + (1.0 - blended_mask) * z_t_orig

        # === 6. 解码回图像 ===
        latents_edit = 1 / pipe.vae.config.scaling_factor * latents_edit
        image = pipe.vae.decode(latents_edit).sample  # (1,3,H,W)
        # 把 [-1,1] -> [0,1]
        image = (image / 2 + 0.5).clamp(0, 1)
        # 转成 PIL
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image
