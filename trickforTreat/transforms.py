"""
trickforTreat/transforms.py

Image -> tensor transforms. The eval transform is what runtime inference uses; the train
transform adds augmentations aimed at the real deployment (window light, ESP32 blur/JPEG).
Crops are already square (detector.crop_with_fill), so there is no center crop anywhere.
"""

from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torchvision import transforms as T


def build_eval_transform(input_size: int, mean, std):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class RandomJPEG:
    """Re-encode as JPEG at a random quality, mimicking MJPEG compression artifacts."""

    def __init__(self, p: float, quality=(30, 90)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_train_transform(aug_cfg, input_size: int, mean, std):
    cj = aug_cfg.color_jitter
    k = int(aug_cfg.gaussian_blur_kernel)
    return T.Compose([
        # Light crop/scale jitter (mimics detector box jitter). Scale floor stays high so the dog isn't cut off.
        T.RandomResizedCrop(input_size, scale=tuple(aug_cfg.random_resized_crop_scale),
                            ratio=tuple(aug_cfg.random_resized_crop_ratio), antialias=True),
        T.RandomHorizontalFlip(p=float(aug_cfg.hflip_p)),  # posture is left/right symmetric; no vertical flips
        T.RandomApply([T.ColorJitter(cj.brightness, cj.contrast, cj.saturation, cj.hue)],
                      p=float(aug_cfg.color_jitter_p)),  # window light changes
        T.RandomGrayscale(p=float(aug_cfg.grayscale_p)),
        T.RandomApply([T.GaussianBlur(k, tuple(aug_cfg.gaussian_blur_sigma))], p=float(aug_cfg.gaussian_blur_p)),
        RandomJPEG(float(aug_cfg.jpeg_compression_p)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def denormalize(x: torch.Tensor, mean, std) -> Image.Image:
    """Normalized CHW tensor -> PIL image, for looking at augmented samples."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return T.ToPILImage()((x * s + m).clamp(0, 1))
