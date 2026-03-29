# diffusion/diffedit.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

from diffusers import StableDiffusionPipeline


@dataclass
class DiffEditConfig:
    # 掩膜生成步数，略少于最终编辑的步数也行
    num_inference_steps: int = 30
    # 随机种子，控制复现
    seed: int = 1234
    # 掩膜阈值（0~1），可以后面调
    thresh: float = 0.5


class DiffEditMaskGenerator:
    """
    基于 Stable Diffusion 实现 DiffEdit 风格的自动掩膜：
    - 源路径：原图 z0 + 同一份噪声 → 前向加噪 z_t^src
    - 目标路径：纯噪声起步 + prompt_tgt → 反向生成 z_t^tgt
    - 在每个时间步计算 || z_t^src - z_t^tgt ||，按时间做平均，归一化 → heatmap
    - 阈值化 → 二值掩膜
    """

    def __init__(
        self,
        sd_model: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.device = device
        # 使用 dtype 而不是 torch_dtype 以避免废弃警告
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            dtype=torch_dtype,
        ).to(device)

    # ------- 一些辅助编码函数 -------

    @torch.no_grad()
    def encode_image(self, image_pil) -> torch.Tensor:
        """
        将 PIL Image 编码到 VAE latent 空间: (1, 4, H/8, W/8)
        """
        pipe = self.pipe

        # 用 VaeImageProcessor 的 preprocess，而不是把 image_processor 当函数调用
        image = pipe.image_processor.preprocess(image_pil)  # (1,3,H,W)

        # 放到正确的 device 和 dtype（与 UNet 保持一致）
        image = image.to(self.device, dtype=pipe.unet.dtype)

        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        获取目标提示的 text embedding
        返回 (1, 77, hidden_dim)

        这里不再用私有 _encode_prompt，而是直接用新的 encode_prompt。
        新版 diffusers 里 encode_prompt 可能返回：
          - Tensor: prompt_embeds
          - (prompt_embeds, negative_prompt_embeds) 这样的 tuple
        我们只取正向的那个就够了。
        """
        pipe = self.pipe

        res = pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,  # 我们这里不做 CFG
        )

        if isinstance(res, tuple):
            # (prompt_embeds, negative_prompt_embeds)
            prompt_embeds = res[0]
        else:
            prompt_embeds = res

        return prompt_embeds  # (1,77,hidden)

    # ------- DiffEdit 掩膜核心 -------

    @torch.no_grad()
    def compute_diffedit_mask(
        self,
        image_pil,
        prompt_src: str,
        prompt_tgt: str,
        cfg: Optional[DiffEditConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          mask_latent_bin: (h, w) 0/1, latent 空间二值掩膜（给 Blended Diffusion 用）
          mask_img_bin:    (H, W) 0/1, 图像分辨率二值掩膜（对比 YOLO/SAM2 用）
          heat_img:        (H, W) [0,1] 连续热力图（可视化用）
        """
        if cfg is None:
            cfg = DiffEditConfig()

        pipe = self.pipe
        device = self.device
        scheduler = pipe.scheduler

        # 1) 编码原图到 latent z0
        z0 = self.encode_image(image_pil)  # (1,4,h,w)
        b, c, h, w = z0.shape
        H, W = image_pil.height, image_pil.width

        # 2) 设置时间步
        scheduler.set_timesteps(cfg.num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # 3) 固定随机噪声（用全局 seed，避免 randn_like(generator=...) 的兼容问题）
        torch.manual_seed(cfg.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(cfg.seed)

        noise = torch.randn(
            z0.shape,
            device=z0.device,
            dtype=z0.dtype,
        )

        # 源路径：z_t^src = q(z_t | z0)
        src_latents_seq = []
        for t in timesteps:
            z_t_src = scheduler.add_noise(z0, noise, t)
            src_latents_seq.append(z_t_src)

        # 4) 目标路径：从纯噪声 + prompt_tgt 反向生成 z_t^tgt
        text_embeds_tgt = self.encode_prompt(prompt_tgt)  # (1,77,hidden)

        # 初始化噪声 latent（和 pipeline 默认方式一致）
        latents_tgt = noise * scheduler.init_noise_sigma
        tgt_latents_seq = []

        for t in timesteps:
            # 这里 batch 维度都是 1，和 text_embeds_tgt 一致
            latent_model_input = scheduler.scale_model_input(latents_tgt, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds_tgt,
            ).sample  # (1,4,h,w)

            # 标准一步 denoising
            latents_tgt = scheduler.step(
                noise_pred, t, latents_tgt
            ).prev_sample  # (1,4,h,w)

            tgt_latents_seq.append(latents_tgt.clone())

        # 5) 计算每个时间步的差异 delta_t = || z_t^src - z_t^tgt ||_2
        deltas = []
        for z_src, z_tgt in zip(src_latents_seq, tgt_latents_seq):
            # (1,4,h,w) -> (1,1,h,w)
            delta = (z_src - z_tgt).pow(2).sum(dim=1, keepdim=True).sqrt()
            deltas.append(delta)

        # 堆叠后在时间维度做平均
        delta_stack = torch.stack(deltas, dim=0)  # (T,1,h,w)
        delta_agg = delta_stack.mean(dim=0)       # (1,1,h,w)

        # 6) 归一化到 [0,1]
        delta_agg = delta_agg[0, 0]  # (h,w)
        d_min = delta_agg.min()
        d_max = delta_agg.max()
        heat_latent = (delta_agg - d_min) / (d_max - d_min + 1e-8)  # (h,w)

        # 7) 阈值生成 latent 掩膜
        mask_latent_bin = (heat_latent >= cfg.thresh).float()       # (h,w)

        # 8) 上采样到图像分辨率，方便和 YOLO/SAM2 对比 / 可视化
        heat_img = F.interpolate(
            heat_latent.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # (H,W)

        mask_img_bin = (heat_img >= cfg.thresh).float()             # (H,W)

        return mask_latent_bin, mask_img_bin, heat_img
