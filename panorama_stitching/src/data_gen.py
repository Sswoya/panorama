from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
S = 2  # 超采样倍数，画完再降采样，相当于免费抗锯齿


# --------------------------------------------------------------------------- #
# 相机模型
# --------------------------------------------------------------------------- #
def camera_matrix(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 Nx2 点做单应变换。"""
    p = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


# --------------------------------------------------------------------------- #
# 场景渲染
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _vertical_gradient(w: int, h: int, top, bottom) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array(top, dtype=np.float32)
    bottom = np.array(bottom, dtype=np.float32)
    img = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
    return np.repeat(img, w, axis=1)


def _add_sun(img: np.ndarray, cx: int, cy: int, radius: int, strength: float) -> None:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    glow = np.exp(-(d ** 2) / (2 * (radius * 0.55) ** 2)).astype(np.float32)
    core = np.clip(1.0 - d / max(radius, 1), 0, 1) ** 2
    g = (0.55 * glow + 0.45 * core)[..., None] * strength
    img += (g * np.array([90, 160, 230], dtype=np.float32)).astype(np.float32)
    np.clip(img, 0, 255, out=img)


def _ridge(rng: np.random.Generator, w: int, base_y: float, amp: float, n: int = 26) -> list[tuple[float, float]]:
    xs = np.linspace(-20, w + 20, n)
    ys = base_y + rng.normal(0, amp, n) * np.sin(np.linspace(0.3, 3.1, n)) ** 2
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    return pts + [(w + 20, base_y + amp * 4), (-20, base_y + amp * 4)]


def _surface_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5, base: int = 6, amp: float = 9.0) -> np.ndarray:
    #多倍频值噪声，模拟真实表面的细微纹理。
    acc = np.zeros((h, w), dtype=np.float32)
    a, tot = 1.0, 0.0
    for o in range(octaves):
        g = max(2, base * (2 ** o))
        grid = rng.random((g, g)).astype(np.float32)
        acc += a * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += a
        a *= 0.55
    acc /= max(tot, 1e-6)
    return (acc - 0.5) * 2.0 * amp


def render_street(w: int, h: int, seed: int = 7) -> np.ndarray:
    """渲染一幅"校园街景"宽幅图，作为全景真值。返回 BGR。"""
    rng = np.random.default_rng(seed)
    W, H = w * S, h * S
    horizon = int(H * 0.575)

    img = _vertical_gradient(W, H, (196, 226, 248), (238, 245, 250))  # RGB 天空
    _add_sun(img, int(W * 0.74), int(H * 0.16), int(H * 0.30), 0.45)

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    #云
    cloud = Image.new("L", (W, H), 0)
    cd = ImageDraw.Draw(cloud)
    for _ in range(26):
        cx, cy = rng.uniform(0, W), rng.uniform(H * 0.03, H * 0.34)
        rw, rh = rng.uniform(W * 0.03, W * 0.09), rng.uniform(H * 0.015, H * 0.04)
        cd.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=int(rng.uniform(40, 110)))
    cloud = cloud.filter(ImageFilter.GaussianBlur(H * 0.012))
    white = Image.new("RGB", (W, H), (255, 255, 255))
    pil = Image.composite(white, pil, cloud.point(lambda v: int(v * 0.85)))
    draw = ImageDraw.Draw(pil, "RGBA")

    #远山两层
    for k, (col, amp, off) in enumerate(
        [((150, 168, 186), H * 0.030, 0.0), ((124, 146, 170), H * 0.045, H * 0.02)]
    ):
        pts = _ridge(rng, W, horizon - H * 0.055 + off, amp)
        draw.polygon([(x, y) for x, y in pts], fill=col + (255,))

    #建筑群
    x = -int(W * 0.02)
    palette = [
        (196, 180, 165), (172, 168, 178), (206, 196, 176),
        (160, 172, 186), (188, 166, 158), (178, 186, 172),
    ]
    signs = ["图书馆", "教学楼 A 座", "实验中心", "咖啡 · CAFE", "计算机学院", "体育馆", "湖工大 1952"]
    si = 0
    while x < W:
        bw = int(rng.uniform(W * 0.045, W * 0.105))
        bh = int(rng.uniform(H * 0.16, H * 0.34))
        top = horizon - bh
        col = palette[int(rng.integers(0, len(palette)))]
        draw.rectangle([x, top, x + bw, horizon + H * 0.02], fill=col + (255,))
        # 楼顶边线 + 女儿墙
        draw.rectangle([x, top, x + bw, top + H * 0.008], fill=tuple(max(0, c - 35) for c in col) + (255,))
        # 窗户
        win_w, win_h = int(bw * 0.13), int(bh * 0.085)
        gap_x, gap_y = int(win_w * 0.95), int(win_h * 1.15)
        wy = top + int(bh * 0.10)
        while wy + win_h < horizon - H * 0.01:
            wx = x + int(bw * 0.08)
            while wx + win_w < x + bw - int(bw * 0.06):
                lit = rng.random()
                if lit < 0.16:
                    c = (255, 236, 188)
                elif lit < 0.30:
                    c = (108, 124, 140)
                else:
                    c = (72, 96, 118)
                draw.rectangle([wx, wy, wx + win_w, wy + win_h], fill=c + (255,))
                draw.line([wx, wy, wx + win_w, wy], fill=(255, 255, 255, 60), width=max(1, S // 2))
                wx += win_w + gap_x
            wy += win_h + gap_y
        # 招牌
        if si < len(signs) and rng.random() < 0.6 and bw > W * 0.055:
            fs = int(bh * 0.075)
            f = _font(max(fs, 10))
            label = signs[si]
            si += 1
            tw = draw.textlength(label, font=f)
            bx = x + (bw - tw) / 2
            by = horizon - bh * 0.045
            draw.rectangle([bx - fs * 0.4, by - fs * 0.35, bx + tw + fs * 0.4, by + fs * 1.1], fill=(238, 236, 230, 235))
            draw.text((bx, by - fs * 0.15), label, font=f, fill=(40, 46, 56, 255))
        # 楼顶细节
        if rng.random() < 0.5:
            ax = x + int(bw * rng.uniform(0.2, 0.8))
            draw.line([ax, top, ax, top - bh * 0.10], fill=(90, 92, 98, 255), width=max(2, S))
        x += bw + int(rng.uniform(W * 0.004, W * 0.012))

    #大气透视
    haze_rgb = Image.new("RGB", (W, H), (214, 228, 242))
    yy = np.arange(H)[:, None].astype(np.float32)
    alpha = np.clip(1.0 - np.abs(horizon - yy) / (H * 0.55), 0, 1) ** 1.6 * 62
    alpha = np.repeat(alpha.astype(np.uint8), W, axis=1)
    pil = Image.composite(haze_rgb, pil, Image.fromarray(alpha))
    draw = ImageDraw.Draw(pil, "RGBA")

    #地面 / 广场
    ground = _vertical_gradient(W, H - horizon, (168, 166, 160), (118, 118, 116))
    pil.paste(Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8)), (0, horizon))
    draw = ImageDraw.Draw(pil, "RGBA")

    #道路
    road_top = int(H * 0.80)
    draw.rectangle([0, road_top, W, int(H * 0.93)], fill=(74, 76, 80, 255))
    dash_y = int(H * 0.865)
    dash_w, gap = int(W * 0.022), int(W * 0.016)
    dx = 0
    while dx < W:
        draw.rectangle([dx, dash_y, dx + dash_w, dash_y + int(H * 0.010)], fill=(232, 228, 205, 235))
        dx += dash_w + gap
    draw.rectangle([0, road_top, W, road_top + int(H * 0.004)], fill=(214, 212, 206, 255))
    draw.rectangle([0, int(H * 0.93), W, int(H * 0.937)], fill=(216, 212, 196, 255))

    #行道树
    trees = []
    tx = rng.uniform(-W * 0.01, W * 1.01)
    while tx < W * 1.02:
        ty = rng.uniform(H * 0.60, H * 0.79)
        depth = (ty - H * 0.60) / (H * 0.19)
        trees.append((tx, ty, 0.45 + 0.9 * depth))
        tx += rng.uniform(W * 0.03, W * 0.075)
    for tx, ty, sc in sorted(trees, key=lambda t: t[1]):
        trunk_h = H * 0.10 * sc
        tw = max(2, int(H * 0.008 * sc * 2))
        draw.rectangle([tx - tw, ty - trunk_h, tx + tw, ty], fill=(92, 74, 58, 255))
        for i in range(4):
            rr = H * 0.055 * sc * (1.0 - 0.14 * i)
            cxx = tx + rng.uniform(-1, 1) * rr * 0.45
            cyy = ty - trunk_h - rr * 0.35 + rng.uniform(-1, 1) * rr * 0.3
            g = int(rng.uniform(-18, 18))
            draw.ellipse(
                [cxx - rr, cyy - rr * 0.85, cxx + rr, cyy + rr * 0.85],
                fill=(int(58 + g), int(104 + g), int(52 + g), 240),
            )

    #车辆
    for _ in range(5):
        cx = rng.uniform(0, W)
        cy = rng.uniform(H * 0.815, H * 0.915)
        sc = 0.7 + 1.0 * (cy - H * 0.815) / (H * 0.10)
        cw, ch = H * 0.075 * sc, H * 0.030 * sc
        body = [tuple(int(v) for v in c) for c in [(198, 62, 58), (54, 92, 160), (232, 232, 230), (48, 50, 56), (214, 168, 60)]]
        c = body[int(rng.integers(0, len(body)))]
        draw.rounded_rectangle([cx, cy - ch, cx + cw, cy], radius=ch * 0.3, fill=c + (255,))
        draw.rounded_rectangle(
            [cx + cw * 0.22, cy - ch * 1.02, cx + cw * 0.74, cy - ch * 0.35],
            radius=ch * 0.22, fill=(150, 190, 210, 255),
        )
        for wx in (cx + cw * 0.18, cx + cw * 0.80):
            draw.ellipse([wx - ch * 0.30, cy - ch * 0.34, wx + ch * 0.30, cy + ch * 0.26], fill=(28, 28, 30, 255))

    #行人
    for _ in range(22):
        px = rng.uniform(0, W)
        py = rng.uniform(H * 0.665, H * 0.90)
        sc = 0.5 + 0.9 * (py - H * 0.665) / (H * 0.235)
        ph = H * 0.032 * sc
        c = tuple(int(v) for v in rng.integers(40, 210, 3))
        draw.ellipse([px - ph * 0.22, py - ph, px + ph * 0.22, py - ph * 0.62], fill=(226, 196, 172, 255))
        draw.rounded_rectangle([px - ph * 0.26, py - ph * 0.62, px + ph * 0.26, py], radius=ph * 0.15, fill=c + (255,))

    #灯杆
    for _ in range(4):
        lx = rng.uniform(0, W)
        ly = rng.uniform(H * 0.78, H * 0.86)
        lh = H * 0.20
        draw.line([lx, ly, lx, ly - lh], fill=(96, 96, 100, 255), width=max(2, S * 2))
        draw.line([lx, ly - lh, lx + H * 0.03, ly - lh], fill=(96, 96, 100, 255), width=max(2, S * 2))

    out = np.asarray(pil).astype(np.float32)

    #颗粒感 + 轻微暗角
    rng2 = np.random.default_rng(seed + 1)
    out += rng2.normal(0, 3.0, out.shape[:2])[..., None]
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    out *= (1.0 - 0.10 * np.clip(r - 0.35, 0, 1.6))[..., None]

    out = np.clip(out, 0, 255).astype(np.uint8)
    small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    # 表面纹理（在 1x 分辨率上叠加，保证每个像素邻域都独一无二）
    tex = _surface_noise(h, w, np.random.default_rng(seed + 31), amp=10.0)[..., None]
    small = np.clip(small.astype(np.float32) + tex, 0, 255).astype(np.uint8)
    return cv2.cvtColor(small, cv2.COLOR_RGB2BGR)


def render_facade(w: int, h: int, seed: int = 21) -> np.ndarray:
    """渲染"重复窗格立面"：专门用来复现重复纹理导致的误匹配。"""
    rng = np.random.default_rng(seed)
    W, H = w * S, h * S
    base = (186, 176, 162)
    img = Image.new("RGB", (W, H), base)
    draw = ImageDraw.Draw(img, "RGBA")

    # 砖缝纹理
    for y in range(0, H, max(6, S * 4)):
        draw.line([0, y, W, y], fill=(170, 160, 148, 90), width=1)

    win_w, win_h = int(W * 0.072), int(H * 0.115)
    gap_x, gap_y = int(W * 0.030), int(H * 0.055)
    y = int(H * 0.05)
    while y + win_h < H:
        x = int(W * 0.03)
        while x + win_w < W:
            draw.rectangle([x - 3, y - 3, x + win_w + 3, y + win_h + 3], fill=(146, 138, 128, 255))
            draw.rectangle([x, y, x + win_w, y + win_h], fill=(58, 78, 96, 255))
            draw.rectangle([x, y, x + win_w, y + int(win_h * 0.16)], fill=(206, 206, 200, 255))
            draw.line([x + win_w // 2, y, x + win_w // 2, y + win_h], fill=(40, 54, 68, 255), width=max(2, S))
            draw.line([x, y + win_h // 2, x + win_w, y + win_h // 2], fill=(40, 54, 68, 255), width=max(2, S))
            x += win_w + gap_x
        y += win_h + gap_y

    # 少量"独有"元素，用来验证算法能否找到正确位置
    f = _font(int(H * 0.05))
    draw.rectangle([W * 0.41, H * 0.02, W * 0.59, H * 0.085], fill=(226, 224, 216, 255))
    draw.text((W * 0.425, H * 0.028), "图书馆 · LIBRARY", font=f, fill=(36, 40, 48, 255))
    draw.line([W * 0.235, 0, W * 0.235, H], fill=(118, 112, 104, 255), width=max(3, S * 3))
    draw.line([W * 0.905, 0, W * 0.905, H], fill=(118, 112, 104, 255), width=max(3, S * 3))
    draw.rectangle([W * 0.02, H * 0.86, W * 0.10, H], fill=(96, 74, 56, 255))

    out = np.asarray(img).astype(np.float32)
    out += rng.normal(0, 3.0, out.shape[:2])[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

@dataclass
class Pair:
    left: np.ndarray
    right: np.ndarray
    H_gt: np.ndarray                      # left -> right 的真值单应
    name: str
    meta: dict = field(default_factory=dict)
    scene: np.ndarray | None = None       # 全景真值（用于定量评价）
    A_left: np.ndarray | None = None      # scene(offset) -> left
    A_right: np.ndarray | None = None     # scene(offset) -> right

    def save(self, directory) -> None:
        import os

        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(os.path.join(directory, f"{self.name}_left.png"), self.left)
        cv2.imwrite(os.path.join(directory, f"{self.name}_right.png"), self.right)


@dataclass
class Camera:
    f: float = 2000.0
    width: int = 1700
    height: int = 1100
    yaw1: float = -11.5
    yaw2: float = 11.5
    pitch: float = 1.5
    roll: float = 0.0


def make_pair(
    scene_fn,
    cam: Camera,
    name: str,
    seed: int = 0,
    gain: float = 0.80,
    color_cast=(1.0, 0.97, 1.06),
    noise_sigma: float = 2.0,
    blur: float = 0.0,
    scene_seed: int = 7,
) -> Pair:
    """渲染场景 + 两次拍摄，返回图对与真值 H。"""
    K = camera_matrix(cam.f, cam.width, cam.height)
    Kinv = np.linalg.inv(K)
    R1 = rot_z(cam.roll) @ rot_x(cam.pitch) @ rot_y(cam.yaw1)
    R2 = rot_z(cam.roll) @ rot_x(cam.pitch) @ rot_y(cam.yaw2)
    A1 = K @ R1 @ Kinv
    A2 = K @ R2 @ Kinv

    # 两张视图投影回场景平面后的联合包围盒，决定场景要画多大
    cw, ch = cam.width, cam.height
    corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64)
    pts = np.vstack([apply_h(np.linalg.inv(A1), corners), apply_h(np.linalg.inv(A2), corners)])
    pad = 24
    x0, y0 = np.floor(pts.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + pad
    sw, sh = int(x1 - x0), int(y1 - y0)

    scene = scene_fn(sw, sh, scene_seed)

    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    A1o = A1 @ np.linalg.inv(T)
    A2o = A2 @ np.linalg.inv(T)

    left = cv2.warpPerspective(scene, A1o, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right = cv2.warpPerspective(scene, A2o, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 第二张图：曝光偏暗 + 色偏 + 噪声 + 轻微运动模糊
    r = right.astype(np.float32) * gain
    for c in range(3):
        r[..., c] *= color_cast[c]
    if blur > 0:
        k = int(blur * 2) * 2 + 1
        r = cv2.GaussianBlur(r, (k, k), 0)
        M = np.float32([[1, 0, blur * 0.4], [0, 1, blur * 0.2]])
        r = cv2.warpAffine(r, M, (cw, ch), flags=cv2.INTER_LINEAR)
    rng = np.random.default_rng(seed + 99)
    r += rng.normal(0, noise_sigma, r.shape)
    right = np.clip(r, 0, 255).astype(np.uint8)

    H_gt = A2o @ np.linalg.inv(A1o)
    H_gt /= H_gt[2, 2]

    ov = overlap_ratio(A1o, A2o, cw, ch)
    meta = {
        "scene_size": (sw, sh),
        "yaw_sep": abs(cam.yaw2 - cam.yaw1),
        "overlap": ov,
        "gain": gain,
        "color_cast": list(color_cast),
        "noise_sigma": noise_sigma,
        "blur": blur,
    }
    return Pair(
        left=left,
        right=right,
        H_gt=H_gt,
        name=name,
        meta=meta,
        scene=scene,
        A_left=A1o,
        A_right=A2o,
    )


def overlap_ratio(A1: np.ndarray, A2: np.ndarray, w: int, h: int) -> float:
    """计算两张视图的重叠率（重叠面积 / 较小面积）。"""
    inv1, inv2 = np.linalg.inv(A1), np.linalg.inv(A2)
    p1 = apply_h(inv1, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    p2 = apply_h(inv2, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    a = cv2.contourArea(p1.astype(np.float32))
    b = cv2.contourArea(p2.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(p1.astype(np.float32), p2.astype(np.float32))
    return float(inter / max(min(a, b), 1e-6))
