# scripts/graph_delta/sd_inpaint.py
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
    _DIFFUSERS_AVAILABLE = True
except Exception:
    StableDiffusionInpaintPipeline = None
    StableDiffusionXLInpaintPipeline = None
    _DIFFUSERS_AVAILABLE = False

try:
    import cv2
except Exception:
    cv2 = None

from .mask_ops import _clamp_bbox_xyxy


def _has_any_safetensors(model_dir: str) -> bool:
    for dp, _, fs in os.walk(model_dir):
        for f in fs:
            if f.endswith(".safetensors"):
                return True
    return False


class SDInpaintEditor:
    """SD1.5 + SDXL inpaint 封装"""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        prefer_safetensors: bool = True,
    ):
        if not _DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not available")
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_id = model_id
        self.prefer_safetensors = bool(prefer_safetensors)

        self.pipe = self._load_pipe(model_id)

        try:
            self.pipe.set_progress_bar_config(disable=False)
        except Exception:
            pass
        if device.type == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

    def _pipe_cls(self, model_id: str):
        name = os.path.basename(os.path.abspath(model_id)).lower()
        is_sdxl = ("sdxl" in name) or ("stable-diffusion-xl" in name) or ("xl" in name)
        return StableDiffusionXLInpaintPipeline if is_sdxl else StableDiffusionInpaintPipeline

    def _load_pipe(self, model_id: str):
        pipe_cls = self._pipe_cls(model_id)

        # 1) try safetensors first
        if self.prefer_safetensors:
            if not _has_any_safetensors(model_id):
                # 只是提示，不中断；有些模型目录确实没有 safetensors
                print(f"[WARN] no .safetensors found under {model_id}. will try anyway.", flush=True)
            try:
                pipe = pipe_cls.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    use_safetensors=True,
                ).to(self.device)
                return pipe
            except Exception as e:
                print(f"[WARN] load safetensors failed: {e}", flush=True)

        # 2) fallback to .bin
        pipe = pipe_cls.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            use_safetensors=False,
        ).to(self.device)
        return pipe

    @torch.inference_mode()
    def run_inpaint(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.5,
        seed: int = 123,
        strength: float = 0.85,
    ) -> Image.Image:
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gen,
            strength=float(strength),
        )
        img_out = out.images[0]
        if img_out.size != image_pil.size:
            img_out = img_out.resize(image_pil.size, resample=Image.LANCZOS)
        return img_out

    @torch.inference_mode()
    def run_inpaint_crop(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.5,
        seed: int = 123,
        strength: float = 0.85,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None,
        blend_blur: int = 11,
    ) -> Image.Image:
        W, H = image_pil.size
        if crop_bbox is None:
            return self.run_inpaint(
                image_pil=image_pil,
                mask_pil=mask_pil,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                strength=strength,
            )

        x0, y0, x1, y1 = _clamp_bbox_xyxy(crop_bbox, W, H)
        img_c = image_pil.crop((x0, y0, x1, y1)).convert("RGB")
        m_c = mask_pil.crop((x0, y0, x1, y1)).convert("L")

        out_c = self.run_inpaint(
            image_pil=img_c,
            mask_pil=m_c,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            strength=strength,
        )

        base_arr = np.array(image_pil.convert("RGB")).astype(np.float32)
        out_arr = np.array(out_c.convert("RGB")).astype(np.float32)
        m = np.array(m_c.convert("L")).astype(np.float32) / 255.0

        m_bin = (m > 0.5).astype(np.float32)
        if cv2 is not None and blend_blur and blend_blur > 0:
            k = int(blend_blur)
            if k % 2 == 0:
                k += 1
            m_feather = cv2.GaussianBlur(m_bin, (k, k), 0)
        else:
            m_feather = m_bin

        alpha = np.clip(m_feather, 0.0, 1.0)[..., None]
        patch_base = base_arr[y0:y1, x0:x1, :]
        patch_mix = out_arr * alpha + patch_base * (1.0 - alpha)
        base_arr[y0:y1, x0:x1, :] = patch_mix
        return Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8), mode="RGB")


def make_sd_inpaint_editor(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    fallback_model_id: Optional[str] = None,
    prefer_safetensors: bool = True,
) -> SDInpaintEditor:
    """
    工厂函数：统一创建 SDInpaintEditor，并支持失败 fallback。
    """
    if not model_id:
        raise ValueError("model_id is empty")

    try:
        return SDInpaintEditor(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            prefer_safetensors=prefer_safetensors,
        )
    except Exception as e:
        if fallback_model_id and (fallback_model_id != model_id):
            print(f"[WARN] failed to load model: {model_id}\n  -> fallback to: {fallback_model_id}", flush=True)
            return SDInpaintEditor(
                model_id=fallback_model_id,
                device=device,
                torch_dtype=torch_dtype,
                prefer_safetensors=prefer_safetensors,
            )
        raise e
