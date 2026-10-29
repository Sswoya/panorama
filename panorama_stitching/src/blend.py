"""重叠区融合：直接覆盖 / 线性羽化 / 多频段（拉普拉斯金字塔）融合。

真实拍摄的两张图曝光不可能完全一致，直接覆盖会在接缝处留下一条"亮暗分界"。
本模块先做增益补偿把亮度拉齐，再用不同策略做过渡：
    naive     : 后一张直接盖上去，接缝最明显（对照用）
    feather   : 按到边界的距离线性加权，能消除硬边，但宽过渡带会发虚
    multiband : 低频用宽过渡、高频用窄过渡，既不虚也不留缝（Burt & Adelson, 1983）
"""

from __future__ import annotations

import cv2
import numpy as np


def gain_compensation(img_l: np.ndarray, img_r: np.ndarray, mask_l, mask_r, model: str = "linear"):
    overlap = mask_l & mask_r
    if overlap.sum() < 500:
        return img_l, {"gain": [1.0, 1.0, 1.0], "bias": [0.0, 0.0, 0.0]}

    idx = np.flatnonzero(overlap.ravel())
    rng = np.random.default_rng(0)
    if len(idx) > 40000:
        idx = rng.choice(idx, 40000, replace=False)

    l = img_l.reshape(-1, 3)[idx].astype(np.float64)
    r = img_r.reshape(-1, 3)[idx].astype(np.float64)

    gains, biases = [], []
    out = img_l.astype(np.float32).copy()
    flat = out.reshape(-1, 3)
    for c in range(3):
        x, y = l[:, c], r[:, c]          # 左图 -> 右图
        if x.std() < 1e-6:
            a, b = 1.0, 0.0
        elif model == "gain":
            a = float(y.mean() / max(x.mean(), 1e-6))
            b = 0.0
        else:
            A = np.stack([x, np.ones_like(x)], axis=1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(sol[0]), float(sol[1])
            a = float(np.clip(a, 0.5, 2.0))
        gains.append(a)
        biases.append(b)
        flat[:, c] = flat[:, c] * a + b

    out = np.clip(flat.reshape(out.shape), 0, 255)
    return out, {"gain": gains, "bias": biases}


# 权重
def distance_weights(mask_l: np.ndarray, mask_r: np.ndarray, blur: int = 0):
    """用距离变换生成羽化权重：离自己边界越远，权重越大。"""
    dl = cv2.distanceTransform(mask_l.astype(np.uint8), cv2.DIST_L2, 5)
    dr = cv2.distanceTransform(mask_r.astype(np.uint8), cv2.DIST_L2, 5)
    s = dl + dr
    s[s < 1e-6] = 1e-6
    wl = dl / s
    wr = dr / s
    if blur > 0:
        k = int(blur) | 1
        wl = cv2.GaussianBlur(wl, (k, k), 0)
        wr = cv2.GaussianBlur(wr, (k, k), 0)
        tot = np.clip(wl + wr, 1e-6, None)
        wl, wr = wl / tot, wr / tot
    return wl.astype(np.float32), wr.astype(np.float32)

# 三种融合策略
def naive_blend(img_l, img_r, mask_l, mask_r):
    out = img_l.copy()
    out[mask_r] = img_r[mask_r]
    return out


def feather_blend(img_l, img_r, mask_l, mask_r, blur: int = 41):
    wl, wr = distance_weights(mask_l, mask_r, blur=blur)
    out = img_l.astype(np.float32) * wl[..., None] + img_r.astype(np.float32) * wr[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _gauss_pyramid(img: np.ndarray, levels: int):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyramid(img: np.ndarray, levels: int):
    g = _gauss_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(cv2.subtract(g[i], up))
    lap.append(g[-1])
    return lap


def _collapse(lap):
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(lap[i].shape[1], lap[i].shape[0]))
        out = cv2.add(out, lap[i])
    return out


def multiband_blend(img_l, img_r, mask_l, mask_r, levels: int = 5):
    """把图分解成不同频段，每段用不同的过渡带宽度加权后再重建。"""
    wl, wr = distance_weights(mask_l, mask_r, blur=0)
    small = min(img_l.shape[0], img_l.shape[1])
    max_levels = max(1, int(np.log2(max(small, 16))) - 2)
    levels = int(min(levels, max_levels))

    L = img_l.astype(np.float32)
    R = img_r.astype(np.float32)
    lp_l = _lap_pyramid(L, levels)
    lp_r = _lap_pyramid(R, levels)

    w = wl.astype(np.float32)
    gp_w = _gauss_pyramid(w, levels)

    blended = []
    for i in range(levels):
        wi = gp_w[i][..., None]
        blended.append(lp_l[i] * wi + lp_r[i] * (1.0 - wi))
    out = _collapse(blended)

    # 只在有效区域内取值，区域外直接置黑
    valid = (mask_l | mask_r).astype(np.float32)[..., None]
    out = out * valid
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_pair(warped: dict, cfg) -> tuple[np.ndarray, dict]:
    """按配置融合并返回（全景图, 附加信息）。"""
    mask_l, mask_r = warped["mask_left"], warped["mask_right"]
    img_l, img_r = warped["left"], warped["right"]
    info: dict = {}

    if cfg.gain_compensation:
        img_l, info["gain"] = gain_compensation(img_l, img_r, mask_l, mask_r, cfg.exposure_model)

    if cfg.mode == "naive":
        pano = naive_blend(img_l, img_r, mask_l, mask_r)
    elif cfg.mode == "feather":
        pano = feather_blend(img_l, img_r, mask_l, mask_r)
    elif cfg.mode == "multiband":
        pano = multiband_blend(img_l, img_r, mask_l, mask_r, cfg.levels)
    else:
        raise ValueError(f"未知融合模式: {cfg.mode}")

    info["mode"] = cfg.mode
    return np.clip(pano, 0, 255).astype(np.uint8), info
