from __future__ import annotations

import cv2
import numpy as np

from .homography import apply_h


def compute_canvas(H: np.ndarray, wl: int, hl: int, wr: int, hr: int):
    """返回 (T, (W, H))：T 把 H 的投影结果平移到第一象限。"""
    corners_l = apply_h(H, np.array([[0, 0], [wl, 0], [wl, hl], [0, hl]], dtype=np.float64))
    corners_r = np.array([[0, 0], [wr, 0], [wr, hr], [0, hr]], dtype=np.float64)
    pts = np.vstack([corners_l, corners_r])
    x0, y0 = np.floor(pts.min(axis=0))
    x1, y1 = np.ceil(pts.max(axis=0))
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    size = (int(x1 - x0), int(y1 - y0))
    return T, size, (int(x0), int(y0))


def warp_to_canvas(img: np.ndarray, M: np.ndarray, size: tuple[int, int]):
    """把 img 按 M 变换到 size 画布，同时返回有效区域掩码。"""
    warped = cv2.warpPerspective(
        img, M, size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ones = np.ones(img.shape[:2], dtype=np.uint8) * 255
    mask = cv2.warpPerspective(
        ones, M, size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask


def warp_pair(left: np.ndarray, right: np.ndarray, H: np.ndarray):
    """左图用 H 投影、右图只做平移，返回两张画布图 + 掩码 + 画布参数。"""
    hl, wl = left.shape[:2]
    hr, wr = right.shape[:2]
    T, size, offset = compute_canvas(H, wl, hl, wr, hr)

    M_left = T @ H
    warped_l, mask_l = warp_to_canvas(left, M_left, size)
    warped_r, mask_r = warp_to_canvas(right, T, size)
    return {
        "left": warped_l,
        "right": warped_r,
        "mask_left": mask_l > 0,
        "mask_right": mask_r > 0,
        "T": T,
        "M_left": M_left,
        "size": size,
        "offset": offset,
    }


def trim_black(pano: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """裁掉四周全黑的边框。"""
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return pano, valid
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return pano[r0:r1, c0:c1], valid[r0:r1, c0:c1]


def largest_inscribed_rect(mask: np.ndarray, scale: int = 8):
    h, w = mask.shape
    sw, sh = max(1, w // scale), max(1, h // scale)
    small = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 250

    heights = np.zeros(sw + 1, dtype=np.int32)
    best = (0, 0, 0, 0, 0)  # (area, x, y, rw, rh)
    for y in range(sh):
        heights[:sw] = np.where(small[y], heights[:sw] + 1, 0)
        heights[sw] = 0
        stack: list[tuple[int, int]] = []
        for x in range(sw + 1):
            cur = int(heights[x])
            start = x
            while stack and stack[-1][1] >= cur:
                sx, shh = stack.pop()
                area = shh * (x - sx)
                if area > best[0]:
                    best = (area, sx, y - shh + 1, x - sx, shh)
                start = sx
            stack.append((start, cur))

    _, bx, by, bw, bh = best
    if bw <= 0 or bh <= 0:
        return 0, 0, w, h

    x0, y0 = bx * scale, by * scale
    x1, y1 = min(w, (bx + bw) * scale), min(h, (by + bh) * scale)

    # 把矩形收紧到真正全为 True 的区域
    sub = mask[y0:y1, x0:x1]
    rows = np.where(sub.any(axis=1))[0]
    cols = np.where(sub.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, w, h
    y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
    x0, x1 = x0 + cols[0], x0 + cols[-1] + 1
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def crop_rect(img: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]
