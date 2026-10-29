from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FeatureConfig


@dataclass
class MatchResult:
    kp1: list
    kp2: list
    desc1: np.ndarray
    desc2: np.ndarray
    matches: list[tuple[int, int, float]]   # (idx1, idx2, distance)
    pts1: np.ndarray                        # Nx2 float32
    pts2: np.ndarray
    timings: dict

    def __len__(self) -> int:
        return len(self.matches)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _make_detector(cfg: FeatureConfig):
    name = cfg.detector.upper()
    if name == "SIFT":
        return cv2.SIFT_create(
            nfeatures=cfg.max_features,
            contrastThreshold=cfg.contrast_threshold,
        )
    if name == "ORB":
        return cv2.ORB_create(nfeatures=cfg.max_features or 5000)
    # OpenCV 5 的 headless 构建里已经不带 KAZE/AKAZE/BRISK 了，这里给出明确提示
    factory = getattr(cv2, f"{name}_create", None)
    if factory is not None:
        return factory()
    raise ValueError(f"当前 OpenCV 构建不支持检测器 {cfg.detector}（可用：SIFT / ORB）")


def _norm_type(cfg: FeatureConfig) -> int:
    return cv2.NORM_HAMMING if cfg.detector.upper() in ("ORB", "AKAZE", "BRISK") else cv2.NORM_L2


def _grid_nms(kps, desc, min_dist: float):
    """网格化关键点筛选：每个网格只保留响应最强的点，避免特征扎堆。"""
    if min_dist <= 0 or not kps:
        return kps, desc
    best: dict[tuple[int, int], int] = {}
    for i, kp in enumerate(kps):
        key = (int(kp.pt[0] // min_dist), int(kp.pt[1] // min_dist))
        if key not in best or kp.response > kps[best[key]].response:
            best[key] = i
    idx = sorted(best.values())
    keep = np.zeros(len(kps), dtype=bool)
    keep[idx] = True
    desc = desc[keep] if desc is not None else desc
    return [kps[i] for i in idx], desc


def detect(img: np.ndarray, cfg: FeatureConfig) -> tuple[list, np.ndarray, float]:
    import time

    t0 = time.perf_counter()
    gray = to_gray(img)
    if cfg.use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    det = _make_detector(cfg)
    kps, desc = det.detectAndCompute(gray, None)
    kps, desc = _grid_nms(kps, desc, cfg.nms_radius)
    return kps, desc, time.perf_counter() - t0


def match(img1: np.ndarray, img2: np.ndarray, cfg: FeatureConfig) -> MatchResult:
    import time

    t0 = time.perf_counter()
    kp1, d1, t_det1 = detect(img1, cfg)
    kp2, d2, t_det2 = detect(img2, cfg)

    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        empty = np.zeros((0, 2), np.float32)
        return MatchResult(kp1, kp2, d1, d2, [], empty, empty, {"detect": t_det1 + t_det2, "match": 0.0})

    # KNN + 比率检验
    bf = cv2.BFMatcher(_norm_type(cfg))
    knn = bf.knnMatch(d1, d2, k=2)

    good: list[tuple[int, int, float]] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < cfg.ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx, m.distance))

    # 双向一致性：再反着匹配一次，只保留互为最优的配对
    if cfg.cross_check and good:
        rknn = bf.knnMatch(d2, d1, k=2)
        rev: dict[int, int] = {}
        for pair in rknn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < cfg.ratio * n.distance:
                rev[m.queryIdx] = m.trainIdx
        good = [(i, j, dist) for (i, j, dist) in good if rev.get(j) == i]

    good.sort(key=lambda t: t[2])
    pts1 = np.float32([kp1[i].pt for i, _, _ in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[j].pt for _, j, _ in good]).reshape(-1, 2)
    t_match = time.perf_counter() - t0 - t_det1 - t_det2

    return MatchResult(
        kp1=kp1,
        kp2=kp2,
        desc1=d1,
        desc2=d2,
        matches=good,
        pts1=pts1,
        pts2=pts2,
        timings={"detect": t_det1 + t_det2, "match": max(t_match, 0.0)},
    )
