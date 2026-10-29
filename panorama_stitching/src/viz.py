from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
BG = (255, 255, 255)
INK = (32, 36, 44)
MUTED = (120, 128, 140)
ACCENT = (38, 108, 196)
GOOD = (32, 158, 92)
BAD = (214, 60, 60)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def to_pil(bgr: np.ndarray) -> Image.Image:
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_cv(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def text_cn(img: np.ndarray, text: str, org, size: int = 28, color=INK) -> np.ndarray:
    pil = to_pil(img)
    d = ImageDraw.Draw(pil)
    d.text(org, text, font=_font(size), fill=color)
    return to_cv(pil)


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def hstack_pad(imgs: list[np.ndarray], gap: int = 18, pad: int = 24, bg=BG) -> np.ndarray:
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=bg) for i in imgs]
    h = max(i.shape[0] for i in imgs)
    total = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1) + 2 * pad
    out = np.zeros((h + 2 * pad, total, 3), np.uint8)
    out[:] = bg
    x = pad
    for i in imgs:
        out[pad:pad + i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


def vstack_pad(imgs: list[np.ndarray], gap: int = 18, pad: int = 24, bg=BG) -> np.ndarray:
    w = max(i.shape[1] for i in imgs)
    total = sum(i.shape[0] for i in imgs) + gap * (len(imgs) - 1) + 2 * pad
    out = np.zeros((total, w + 2 * pad, 3), np.uint8)
    out[:] = bg
    y = pad
    for i in imgs:
        out[y:y + i.shape[0], pad:pad + i.shape[1]] = i
        y += i.shape[0] + gap
    return out


def grid(imgs: list[np.ndarray], cols: int = 2, gap: int = 12, pad: int = 20, bg=BG) -> np.ndarray:
    """把若干张图排成 cols 列的网格（每格等宽居中）。"""
    if not imgs:
        raise ValueError("empty")
    cell_w = max(i.shape[1] for i in imgs)
    rows = []
    for start in range(0, len(imgs), cols):
        chunk = imgs[start:start + cols]
        fitted = []
        for im in chunk:
            if im.shape[1] != cell_w:
                im = cv2.resize(im, (cell_w, int(im.shape[0] * cell_w / im.shape[1])),
                                interpolation=cv2.INTER_AREA)
            fitted.append(im)
        while len(fitted) < cols:
            fitted.append(np.full((fitted[0].shape[0], cell_w, 3), bg, np.uint8))
        rows.append(hstack_pad(fitted, gap=gap, pad=0, bg=bg))
    return vstack_pad(rows, gap=gap, pad=pad, bg=bg)


def caption(img: np.ndarray, text: str, size: int = 26, color=INK, pad: int = 18, bg=BG) -> np.ndarray:
    """在图片下方加说明文字（返回新图）。"""
    pil = to_pil(img)
    d = ImageDraw.Draw(pil)
    f = _font(size)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    W = max(pil.width, tw + 2 * pad)
    out = Image.new("RGB", (W, pil.height + th + 2 * pad), bg)
    out.paste(pil, ((W - pil.width) // 2, 0))
    d2 = ImageDraw.Draw(out)
    d2.text(((W - tw) // 2, pil.height + pad // 2), text, font=f, fill=color)
    return to_cv(out)


# 特征匹配图
def fig_matches(left: np.ndarray, right: np.ndarray, match, hres, width: int = 2200) -> np.ndarray:
    """上下两栏：全匹配 vs 内点/外点（绿=内点，红=外点）。"""
    l = fit_width(left, width // 2 - 10)
    r = fit_width(right, width // 2 - 10)
    sx, sy = l.shape[1] / left.shape[1], l.shape[0] / left.shape[0]

    kp1 = [(k.pt[0] * sx, k.pt[1] * sy) for k in match.kp1]
    kp2 = [(k.pt[0] * sx, k.pt[1] * sy) for k in match.kp2]

    def render(only_inliers: bool | None) -> np.ndarray:
        canvas = hstack_pad([l, r], gap=12, pad=8)
        off = l.shape[1] + 12 + 8
        pil = to_pil(canvas)
        d = ImageDraw.Draw(pil)
        mask = hres.inlier_mask if hres is not None else np.ones(len(match.matches), bool)
        for idx, (i, j, _) in enumerate(match.matches):
            is_in = bool(mask[idx]) if idx < len(mask) else True
            if only_inliers is True and not is_in:
                continue
            if only_inliers is False and is_in:
                continue
            col = GOOD if is_in else BAD
            p1 = (kp1[i][0] + 8, kp1[i][1] + 8)
            p2 = (kp2[j][0] + off, kp2[j][1] + 8)
            d.line([p1, p2], fill=col, width=1)
        return to_cv(pil)

    a = render(None)
    b = render(True)
    a = caption(a, f"（a）Lowe 比率检验后的初始匹配：{len(match.matches)} 对", size=30)
    n_in = int(hres.inlier_mask.sum()) if hres is not None else 0
    b = caption(b, f"（b）经过 RANSAC 剔除误匹配：绿色内点 {n_in} 对，红色为被剔除的外点", size=30)
    return vstack_pad([a, b], gap=10)


# 投影对位图
def fig_overlay(warped: dict, max_w: int = 1500) -> np.ndarray:
    """把左图投到右图坐标系后，用红/绿通道叠加，对得准 → 只有灰色重合区带彩色边。"""
    l = warped["left"].astype(np.float32)
    r = warped["right"].astype(np.float32)
    gl = cv2.cvtColor(l.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    both = warped["mask_left"] & warped["mask_right"]
    # 只保留重叠区，做成红青叠加
    img = np.zeros((*gl.shape, 3), np.uint8)
    img[..., 2] = gl          # 红 = 左图
    img[..., 1] = gr          # 绿 = 右图
    img[~both] = (img[~both] * 0.35).astype(np.uint8)
    return fit_width(img, max_w)


# 融合方式对比
def fig_blend_compare(pano_dict: dict[str, np.ndarray], rect, zoom_w: int = 420) -> np.ndarray:
    x, y, w, h = rect
    cx = x + w // 2
    rows = []
    for name, img in pano_dict.items():
        crop = img[max(0, y):y + h, max(0, cx - zoom_w // 2):cx + zoom_w // 2]
        crop = fit_width(crop, 900)
        rows.append(caption(crop, name, size=28))
    return vstack_pad(rows, gap=8)


# 流程图
def fig_pipeline(out_path: str, cfg) -> None:
    W, H = 2000, 1120
    img = Image.new("RGB", (W, H), (250, 251, 253))
    d = ImageDraw.Draw(img)
    f_title = _font(38)
    f_box = _font(28)
    f_sub = _font(21)
    f_small = _font(19)

    d.text((60, 36), "图像拼接（全景图生成）整体流程", font=f_title, fill=INK)

    boxes = [
        ("① 输入", ["左图 I_L", "右图 I_R"], 60, 130, 300, 150, (232, 240, 252), ACCENT),
        ("② 预处理", ["灰度化", "CLAHE 均衡", "（可选）去噪"], 420, 130, 300, 200, (236, 246, 240), GOOD),
        ("③ 特征检测", [f"{cfg.features.detector} 关键点", "128/256 维描述子"], 780, 130, 300, 170, (255, 245, 228), (198, 128, 24)),
        ("④ 特征匹配", ["KNN (k=2)", f"Lowe 比率 r={cfg.features.ratio}", "双向一致性校验"], 1140, 130, 340, 220, (243, 236, 252), (128, 72, 190)),
        ("⑤ 单应估计", ["RANSAC 采样 4 点", "归一化 DLT", f"内点阈值 {cfg.ransac.threshold}px", "IRLS 精化"], 1540, 130, 380, 250, (252, 236, 238), BAD),
        ("⑥ 图像投影", ["计算画布包围盒", "M = T · H", "warpPerspective"], 1540, 470, 380, 190, (232, 244, 250), (24, 120, 168)),
        ("⑦ 图像融合", ["增益补偿", f"{cfg.blend.mode} 融合", "拉普拉斯金字塔"], 1080, 470, 380, 190, (238, 246, 232), (72, 148, 60)),
        ("⑧ 输出", ["去除黑边（最大内接矩形）", "全景图 + 评价指标"], 620, 470, 380, 160, (240, 240, 244), MUTED),
    ]

    for title, subs, x, y, w, h, fill, edge in boxes:
        d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=fill, outline=edge, width=3)
        d.text((x + 20, y + 16), title, font=f_box, fill=edge)
        for i, s in enumerate(subs):
            d.text((x + 22, y + 64 + i * 34), "· " + s, font=f_small, fill=(70, 76, 86))

    def arrow(x1, y1, x2, y2):
        d.line([x1, y1, x2, y2], fill=(120, 128, 140), width=3)
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        L = 14
        for s in (+1, -1):
            d.line([x2, y2, x2 - L * math.cos(ang + s * 0.5), y2 - L * math.sin(ang + s * 0.5)],
                   fill=(120, 128, 140), width=3)

    arrow(360, 205, 415, 205)
    arrow(720, 205, 775, 205)
    arrow(1080, 205, 1135, 205)
    arrow(1480, 205, 1535, 205)
    arrow(1730, 380, 1730, 465)
    arrow(1535, 565, 1465, 565)
    arrow(1080, 565, 1005, 565)
    d.line([810, 630, 810, 700], fill=(120, 128, 140), width=3)
    d.text((60, 760), "判据：匹配数 ≥ 8 且内点率 ≥ 10%，否则直接拒绝拼接并报错；单应矩阵另做行列式/投影范围健全性检查。",
           font=f_sub, fill=(80, 88, 100))
    d.text((60, 810), f"融合策略可选：naive（直接覆盖）/ feather（距离加权羽化）/ multiband（多频段，默认）。",
           font=f_sub, fill=(80, 88, 100))
    d.text((60, 860), "特征检测器可选：SIFT（默认，精度最高）/ ORB（最快）/ AKAZE（折中）。单应求解器可选：手写 RANSAC 或 OpenCV。",
           font=f_sub, fill=(80, 88, 100))
    d.text((60, 940), "输出：outputs/*.png 可视化结果、metrics.json 量化指标、report/ 报告文档。",
           font=f_small, fill=(110, 118, 130))

    img.save(out_path)
