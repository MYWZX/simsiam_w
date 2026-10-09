# 简单显著图，用于扰动区域选择
from __future__ import annotations

import cv2
import numpy as np


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-6)


def compute_simple_saliency(image_bgr: np.ndarray) -> np.ndarray:
    """Lightweight saliency fallback.

    Uses gradient magnitude + center prior, avoiding external saliency models.
    Output range: [0, 1], shape (H, W).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = cv2.GaussianBlur(grad, (7, 7), 0)
    grad = normalize_map(grad)

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    sy, sx = h / 3.0, w / 3.0
    center = np.exp(-(((yy - cy) ** 2) / (2 * sy * sy + 1e-6) + ((xx - cx) ** 2) / (2 * sx * sx + 1e-6)))
    center = normalize_map(center)

    sal = 0.7 * grad + 0.3 * center
    sal = cv2.GaussianBlur(sal, (11, 11), 0)
    return normalize_map(sal)
