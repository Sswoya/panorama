from __future__ import annotations

import cv2
import numpy as np


def _to_gray_f(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return img.astype(np.float64)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() == 0:
            return float("nan")
        mse = d[m].mean()
    else:
        mse = d.mean()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def _gaussian_kernel(sigma: float = 1.5, truncate: float = 3.5) -> np.ndarray:
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, sigma: float = 1.5) -> float:
    """结构相似度（高斯加权，11x11 窗口的等价实现）。"""
    ga = _to_gray_f(a)
    gb = _to_gray_f(b)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(ga), blur(gb)
    saa = blur(ga * ga) - mu_a ** 2
    sbb = blur(gb * gb) - mu_b ** 2
    sab = blur(ga * gb) - mu_a * mu_b

    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    m = num / np.maximum(den, 1e-12)

    if mask is not None:
        mm = mask.astype(bool)
        if mm.sum() == 0:
            return float("nan")
        # 腐蚀一点，避开拼接边界处的黑边干扰
        mm = cv2.erode(mm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if mm.sum() == 0:
            return float("nan")
        return float(m[mm].mean())
    return float(m.mean())


def seam_energy(pano: np.ndarray, mask_l: np.ndarray, mask_r: np.ndarray) -> float:
    overlap = (mask_l & mask_r).astype(np.uint8)
    if overlap.sum() == 0:
        return float("nan")
    k = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(overlap, k, iterations=1) - cv2.erode(overlap, k, iterations=2)
    band = edge > 0
    if band.sum() < 50:
        return float("nan")
    g = _to_gray_f(pano)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag[band].mean())


def coverage(pano: np.ndarray, valid: np.ndarray) -> float:
    return float(valid.mean())
