from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HomographyResult:
    H: np.ndarray
    inlier_mask: np.ndarray
    n_inliers: int
    n_iters: int
    time: float
    method: str
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 每个匹配的对称转移误差

    @property
    def inlier_ratio(self) -> float:
        n = len(self.inlier_mask)
        return float(self.n_inliers / n) if n else 0.0

    @property
    def rms_error(self) -> float:
        if self.n_inliers == 0:
            return float("inf")
        return float(np.sqrt(np.mean(self.errors[self.inlier_mask] ** 2)))


# 基础工具
def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float64)
    q = p @ H.T
    w = q[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return q[:, :2] / w


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]], dtype=np.float64)
    return (pts - c) * s, T


def _null_vector(A: np.ndarray) -> np.ndarray:
    G = A.T @ A
    vals, vecs = np.linalg.eigh(G)
    return vecs[:, int(np.argmin(vals))]


def dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = len(src)
    if n < 4:
        raise ValueError("至少需要 4 组点")
    s, Ts = normalize_points(src)
    d, Td = normalize_points(dst)

    A = np.zeros((2 * n, 9), dtype=np.float64)
    xs, ys = s[:, 0], s[:, 1]
    xd, yd = d[:, 0], d[:, 1]
    A[0::2, 0] = -xs
    A[0::2, 1] = -ys
    A[0::2, 2] = -1.0
    A[0::2, 6] = xs * xd
    A[0::2, 7] = ys * xd
    A[0::2, 8] = xd
    A[1::2, 3] = -xs
    A[1::2, 4] = -ys
    A[1::2, 5] = -1.0
    A[1::2, 6] = xs * yd
    A[1::2, 7] = ys * yd
    A[1::2, 8] = yd

    Hn = _null_vector(A).reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12:
        H = H / max(np.abs(H).max(), 1e-12)
    else:
        H = H / H[2, 2]
    return H


def symmetric_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    p = apply_h(H, src)
    q = apply_h(Hinv, dst)
    e1 = np.linalg.norm(p - dst, axis=1)
    e2 = np.linalg.norm(q - src, axis=1)
    return np.maximum(e1, e2)


# --------------------------------------------------------------------------- #
# RANSAC
# --------------------------------------------------------------------------- #
def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 3.0,
    max_iters: int = 5000,
    confidence: float = 0.999,
    seed: int = 0,
    refine_iters: int = 5,
    sampler: str = "prosac",
) -> HomographyResult:

    t0 = time.perf_counter()
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        return HomographyResult(np.eye(3), np.zeros(n, bool), 0, 0, time.perf_counter() - t0, "custom")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_H = np.eye(3)
    best_count = 0
    need = max_iters
    it = 0

    while it < min(need, max_iters):
        it += 1
        if sampler == "prosac" and n > 12:
            # 从小到大放开候选池，前期只用最可信的匹配
            pool = min(n, 10 + int((n - 10) * (it / max_iters) ** 0.7))
            pool = max(pool, 8)
            idx = rng.choice(pool, 4, replace=False)
        else:
            idx = rng.choice(n, 4, replace=False)
        try:
            H = dlt(src[idx], dst[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(H)):
            continue
        err = symmetric_error(H, src, dst)
        mask = err <= threshold
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask, best_H = c, mask, H
            # 自适应迭代次数：内点率 w 越高，需要的采样次数越少
            w = c / n
            denom = np.log(max(1e-12, 1.0 - w ** 4))
            need = int(np.log(1.0 - confidence) / denom) if denom < -1e-9 else max_iters
            need = int(min(max(need, 50), max_iters))

    if best_count >= 4:
        # 1) 用全部内点重估
        best_H = dlt(src[best_mask], dst[best_mask])
        if refine_iters > 0:
            # 2) 迭代重加权：误差小的点权重大，进一步压低重投影误差
            sigma = max(threshold / 2.0, 1e-3)
            for _ in range(refine_iters):
                err = symmetric_error(best_H, src[best_mask], dst[best_mask])
                w = 1.0 / (1.0 + (err / sigma) ** 2)
                if w.sum() < 4:
                    break
                best_H = weighted_dlt(src[best_mask], dst[best_mask], w)
        err = symmetric_error(best_H, src, dst)
        best_mask = err <= threshold
        best_count = int(best_mask.sum())
        best_H = dlt(src[best_mask], dst[best_mask])
        errors = symmetric_error(best_H, src, dst)
    else:
        errors = symmetric_error(best_H, src, dst)

    return HomographyResult(
        H=best_H,
        inlier_mask=best_mask,
        n_inliers=best_count,
        n_iters=it,
        time=time.perf_counter() - t0,
        method="custom",
        errors=errors,
    )


def weighted_dlt(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    s, Ts = normalize_points(src)
    d, Td = normalize_points(dst)
    sw = np.sqrt(np.clip(w, 0, None))[:, None]
    A = np.zeros((2 * len(s), 9))
    xs, ys = s[:, 0], s[:, 1]
    xd, yd = d[:, 0], d[:, 1]
    A[0::2, 0] = -xs
    A[0::2, 1] = -ys
    A[0::2, 2] = -1.0
    A[0::2, 6] = xs * xd
    A[0::2, 7] = ys * xd
    A[0::2, 8] = xd
    A[1::2, 3] = -xs
    A[1::2, 4] = -ys
    A[1::2, 5] = -1.0
    A[1::2, 6] = xs * yd
    A[1::2, 7] = ys * yd
    A[1::2, 8] = yd
    A *= np.repeat(sw, 2, axis=0)
    H = np.linalg.inv(Td) @ _null_vector(A).reshape(3, 3) @ Ts
    return H / H[2, 2]


def opencv_homography(src, dst, threshold: float = 3.0) -> HomographyResult:
    t0 = time.perf_counter()
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold, maxIters=5000, confidence=0.999)
    if H is None:
        H, mask = np.eye(3), np.zeros(len(src), np.uint8)
    mask = mask.ravel().astype(bool)
    errors = symmetric_error(H, np.asarray(src, float), np.asarray(dst, float))
    return HomographyResult(
        H=H,
        inlier_mask=mask,
        n_inliers=int(mask.sum()),
        n_iters=-1,
        time=time.perf_counter() - t0,
        method="opencv",
        errors=errors,
    )


def estimate(src, dst, cfg) -> HomographyResult:
    if cfg.mode == "opencv":
        return opencv_homography(src, dst, cfg.threshold)
    return ransac_homography(
        src,
        dst,
        threshold=cfg.threshold,
        max_iters=cfg.max_iters,
        confidence=cfg.confidence,
        refine_iters=cfg.refine_iters if cfg.refine else 0,
        sampler=cfg.sampler,
    )


def corner_error(H_est: np.ndarray, H_gt: np.ndarray, w: int, h: int) -> float:
    """把左图 4 个角点分别用 H_est / H_gt 投到右图坐标系，求 RMSE（像素）。"""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    a = apply_h(H_est, corners)
    b = apply_h(H_gt, corners)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
