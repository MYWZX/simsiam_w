# 生成 mild / moderate 等扰动图像
from __future__ import annotations

import io
import random
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


PERTURB_WEIGHTS = {
    # AIGC-like artifact perturbations
    "blend_boundary": 0.18,
    "texture_incoherence": 0.16,
    "hf_artifact": 0.12,
    "oversmooth": 0.12,
    "micro_warp": 0.10,
    # classical low-level distortions
    "gaussian_blur": 0.12,
    "jpeg_compression": 0.10,
    "gaussian_noise": 0.06,
    "color_quantization": 0.04,
}

STRENGTH_PRESETS = {
    "mild": {
        "patch_ratio": (0.12, 0.20),
        "num_patches": (1, 1),
        "alpha": 0.45,
        "sigma_blend": 1.5,
        "tex_jitter": 0.05,
        "hf_gain": 0.6,
        "noise_std": 2.0,
        "smooth_sigma": 1.0,
        "warp_px": 2.0,
        "color_shift": 4.0,
        "blur_sigma": 0.9,
        "jpeg_quality": 70,
        "quant_step": 24,
    },
    "moderate": {
        "patch_ratio": (0.18, 0.28),
        "num_patches": (1, 2),
        "alpha": 0.65,
        "sigma_blend": 2.5,
        "tex_jitter": 0.10,
        "hf_gain": 1.0,
        "noise_std": 4.0,
        "smooth_sigma": 1.8,
        "warp_px": 4.0,
        "color_shift": 8.0,
        "blur_sigma": 1.5,
        "jpeg_quality": 45,
        "quant_step": 40,
    },
    "severe": {
        "patch_ratio": (0.24, 0.36),
        "num_patches": (2, 2),
        "alpha": 0.80,
        "sigma_blend": 3.5,
        "tex_jitter": 0.16,
        "hf_gain": 1.4,
        "noise_std": 6.0,
        "smooth_sigma": 2.6,
        "warp_px": 6.0,
        "color_shift": 12.0,
        "blur_sigma": 2.2,
        "jpeg_quality": 25,
        "quant_step": 56,
    },
}


def _normalize_saliency(saliency: np.ndarray, h: int, w: int) -> np.ndarray:
    sal = saliency.astype(np.float32)
    if sal.ndim == 3:
        sal = sal[..., 0]
    if sal.shape != (h, w):
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    sal -= sal.min()
    sal /= (sal.max() + 1e-8)
    return sal


def _random_patch_size(h: int, w: int, ratio_range: Tuple[float, float]) -> Tuple[int, int]:
    base = min(h, w)
    ratio = random.uniform(*ratio_range)
    size = max(16, int(base * ratio))

    aspect = random.uniform(0.75, 1.33)
    ph = int(size / np.sqrt(aspect))
    pw = int(size * np.sqrt(aspect))

    ph = int(np.clip(ph, 16, h))
    pw = int(np.clip(pw, 16, w))
    return ph, pw


def _sample_patch_from_saliency(
    saliency_map: np.ndarray,
    h: int,
    w: int,
    patch_h: int,
    patch_w: int,
    saliency_sampling_ratio: float,
) -> Tuple[int, int, int, int]:
    max_y = max(0, h - patch_h)
    max_x = max(0, w - patch_w)

    if random.random() > saliency_sampling_ratio:
        y1 = random.randint(0, max_y) if max_y > 0 else 0
        x1 = random.randint(0, max_x) if max_x > 0 else 0
        return x1, y1, x1 + patch_w, y1 + patch_h

    ys, xs = np.where(saliency_map > np.percentile(saliency_map, 75))
    if len(xs) == 0:
        y1 = random.randint(0, max_y) if max_y > 0 else 0
        x1 = random.randint(0, max_x) if max_x > 0 else 0
        return x1, y1, x1 + patch_w, y1 + patch_h

    idx = random.randint(0, len(xs) - 1)
    cx, cy = xs[idx], ys[idx]

    x1 = int(np.clip(cx - patch_w // 2, 0, max_x))
    y1 = int(np.clip(cy - patch_h // 2, 0, max_y))
    return x1, y1, x1 + patch_w, y1 + patch_h


def _make_soft_mask(patch_h: int, patch_w: int, sigma: float) -> np.ndarray:
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    margin_y = max(2, patch_h // 8)
    margin_x = max(2, patch_w // 8)
    mask[margin_y:patch_h - margin_y, margin_x:patch_w - margin_x] = 1.0

    k = max(3, int(2 * round(3 * sigma) + 1))
    mask = cv2.GaussianBlur(mask, (k, k), sigmaX=sigma, sigmaY=sigma)
    mask = np.clip(mask, 0.0, 1.0)
    return mask[..., None]


def _blend_patch(
    image: np.ndarray,
    patch: np.ndarray,
    box: Tuple[int, int, int, int],
    alpha_mask: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    region = image[y1:y2, x1:x2].astype(np.float32)
    patch = patch.astype(np.float32)

    out = region * (1.0 - alpha_mask) + patch * alpha_mask

    image_out = image.copy()
    image_out[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)
    return image_out


def _randn_like(shape, scale: float) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32) * scale


def _choose_perturb_type() -> str:
    types = list(PERTURB_WEIGHTS.keys())
    weights = [PERTURB_WEIGHTS[t] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def _apply_blend_boundary(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    sigma = preset["sigma_blend"]
    k = max(3, int(2 * round(3 * sigma) + 1))
    blur = cv2.GaussianBlur(patch, (k, k), sigmaX=sigma, sigmaY=sigma)
    bias = np.array([
        random.uniform(-preset["color_shift"], preset["color_shift"]),
        random.uniform(-preset["color_shift"], preset["color_shift"]),
        random.uniform(-preset["color_shift"], preset["color_shift"]),
    ], dtype=np.float32)
    mixed = 0.65 * patch + 0.35 * blur + bias[None, None, :]
    alpha = _make_soft_mask(patch.shape[0], patch.shape[1], sigma=max(1.0, sigma * 1.2)) * preset["alpha"]
    return _blend_patch(image, mixed, box, alpha)


def _apply_texture_incoherence(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    scale_x = random.uniform(0.85, 1.15)
    scale_y = random.uniform(0.85, 1.15)
    tmp_w = max(8, int(pw * scale_x))
    tmp_h = max(8, int(ph * scale_y))
    tmp = cv2.resize(patch, (tmp_w, tmp_h), interpolation=cv2.INTER_LINEAR)
    tmp = cv2.resize(tmp, (pw, ph), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        sigma_x = random.uniform(0.5, 1.5)
        sigma_y = random.uniform(1.5, 3.0)
    else:
        sigma_x = random.uniform(1.5, 3.0)
        sigma_y = random.uniform(0.5, 1.5)
    tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    tmp = tmp + _randn_like(tmp.shape, preset["tex_jitter"] * 255.0)
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"])) * preset["alpha"]
    return _blend_patch(image, tmp, box, alpha)


def _apply_hf_artifact(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    blur = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.0, sigmaY=1.0)
    detail = patch - blur
    sharpened = patch + preset["hf_gain"] * detail
    sharpened = sharpened + 0.5 * _randn_like(patch.shape, preset["noise_std"])
    second_blur = cv2.GaussianBlur(sharpened, (0, 0), sigmaX=0.6, sigmaY=0.6)
    sharpened = 1.1 * sharpened - 0.1 * second_blur
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"] * 0.8)) * preset["alpha"]
    return _blend_patch(image, sharpened, box, alpha)


def _apply_oversmooth(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    sigma = preset["smooth_sigma"]
    k = max(3, int(2 * round(3 * sigma) + 1))
    sm = cv2.GaussianBlur(patch, (k, k), sigmaX=sigma, sigmaY=sigma)
    sm_u8 = np.clip(sm, 0, 255).astype(np.uint8)
    sm2 = cv2.bilateralFilter(sm_u8, d=5, sigmaColor=20, sigmaSpace=20).astype(np.float32)
    out = 0.4 * patch + 0.6 * sm2
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"])) * preset["alpha"]
    return _blend_patch(image, out, box, alpha)


def _apply_micro_warp(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    amp = preset["warp_px"]
    grid_y, grid_x = np.mgrid[0:ph, 0:pw].astype(np.float32)
    dx = cv2.GaussianBlur(_randn_like((ph, pw), amp), (0, 0), sigmaX=max(ph, pw) / 10.0)
    dy = cv2.GaussianBlur(_randn_like((ph, pw), amp), (0, 0), sigmaX=max(ph, pw) / 10.0)
    map_x = grid_x + dx
    map_y = grid_y + dy
    warped = cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"])) * preset["alpha"]
    return _blend_patch(image, warped, box, alpha)


def _apply_gaussian_blur(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    sigma = preset["blur_sigma"]
    out = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, sigma)) * preset["alpha"]
    return _blend_patch(image, out, box, alpha)


def _jpeg_reencode_bgr(patch_bgr: np.ndarray, quality: int) -> np.ndarray:
    patch_rgb = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(patch_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    rec = np.array(Image.open(buf).convert("RGB"))
    return cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)


def _apply_jpeg_compression(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]
    out = _jpeg_reencode_bgr(patch, preset["jpeg_quality"]).astype(np.float32)
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"] * 0.7)) * preset["alpha"]
    return _blend_patch(image, out, box, alpha)


def _apply_gaussian_noise(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    out = patch + _randn_like(patch.shape, preset["noise_std"] * 1.4)
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"] * 0.7)) * preset["alpha"]
    return _blend_patch(image, out, box, alpha)


def _apply_color_quantization(image: np.ndarray, box: Tuple[int, int, int, int], preset: Dict) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].copy().astype(np.float32)
    ph, pw = patch.shape[:2]
    step = float(preset["quant_step"])
    out = np.round(patch / step) * step
    alpha = _make_soft_mask(ph, pw, sigma=max(1.0, preset["sigma_blend"] * 0.7)) * preset["alpha"]
    return _blend_patch(image, out, box, alpha)


PERTURB_FN_MAP = {
    "blend_boundary": _apply_blend_boundary,
    "texture_incoherence": _apply_texture_incoherence,
    "hf_artifact": _apply_hf_artifact,
    "oversmooth": _apply_oversmooth,
    "micro_warp": _apply_micro_warp,
    "gaussian_blur": _apply_gaussian_blur,
    "jpeg_compression": _apply_jpeg_compression,
    "gaussian_noise": _apply_gaussian_noise,
    "color_quantization": _apply_color_quantization,
}


def perturb_image(
    image_bgr: np.ndarray,
    saliency: np.ndarray,
    strength: str = "mild",
    saliency_sampling_ratio: float = 0.7,
    perturbation_prob: float = 1.0,
):
    if strength not in STRENGTH_PRESETS:
        raise ValueError(f"Unknown strength: {strength}")
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected image_bgr shape [H, W, 3], got {image_bgr.shape}")

    image = image_bgr.copy()
    h, w = image.shape[:2]
    saliency_map = _normalize_saliency(saliency, h, w)

    if random.random() > perturbation_prob:
        meta = {"type": "identity", "strength": strength, "num_patches": 0}
        return image.copy(), meta

    preset = STRENGTH_PRESETS[strength]
    perturb_type = _choose_perturb_type()
    perturb_fn = PERTURB_FN_MAP[perturb_type]
    out = image.copy()
    num_patches = random.randint(*preset["num_patches"])

    for _ in range(num_patches):
        ph, pw = _random_patch_size(h, w, preset["patch_ratio"])
        box = _sample_patch_from_saliency(
            saliency_map=saliency_map,
            h=h,
            w=w,
            patch_h=ph,
            patch_w=pw,
            saliency_sampling_ratio=saliency_sampling_ratio,
        )
        out = perturb_fn(out, box, preset)

    meta = {"type": perturb_type, "strength": strength, "num_patches": num_patches}
    return out, meta
