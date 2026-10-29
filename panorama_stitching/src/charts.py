from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
INK = (36, 40, 48)
MUTED = (130, 138, 150)
GRID = (226, 230, 236)
BG = (255, 255, 255)
SERIES = [
    (38, 108, 196),
    (214, 96, 60),
    (46, 158, 108),
    (150, 96, 190),
    (200, 160, 40),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _nice_max(v: float) -> tuple[float, float]:
    """返回 (上限, 步长)，让刻度落在整齐的数上。"""
    if v <= 0:
        return 1.0, 0.2
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        step = m * base / 4
        top = math.ceil(v / step) * step
        if top / step <= 8:
            return top, step
    return v * 1.1, v * 1.1 / 5


def line_chart(
    out_path: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    size: tuple[int, int] = (1500, 820),
    ymin: float | None = None,
    max_xticks: int = 8,
) -> None:
    """series: [(名字, xs, ys), ...]"""
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_t = _font(34)
    f_l = _font(24)
    f_s = _font(21)

    ml, mr, mt, mb = 120, 240, 90, 110
    plot_w, plot_h = W - ml - mr, H - mt - mb

    all_y = [y for _, _, ys in series for y in ys]
    all_x = [x for _, xs, _ in series for x in xs]
    lo = 0.0 if ymin is None else float(ymin)
    hi = max(all_y) * 1.02
    ystep = _nice_max(hi - lo)[1]
    hi = lo + math.ceil((hi - lo) / ystep) * ystep
    xmin, xmax = min(all_x), max(all_x)

    def px(x):
        return ml + (x - xmin) / max(xmax - xmin, 1e-9) * plot_w

    def py(y):
        return mt + plot_h - (y - lo) / max(hi - lo, 1e-9) * plot_h

    d.text((ml, 28), title, font=f_t, fill=INK)

    steps = int(round((hi - lo) / ystep))
    for i in range(steps + 1):
        y = lo + i * ystep
        yy = py(y)
        d.line([ml, yy, ml + plot_w, yy], fill=GRID, width=1)
        label = f"{y:.0f}" if ystep >= 1 else f"{y:.2f}"
        d.text((ml - 16, yy - 12), label, font=f_s, fill=MUTED, anchor="ra")

    d.line([ml, mt + plot_h, ml + plot_w, mt + plot_h], fill=(170, 176, 188), width=2)
    d.line([ml, mt, ml, mt + plot_h], fill=(170, 176, 188), width=2)

    for si, (name, xs_, ys) in enumerate(series):
        col = SERIES[si % len(SERIES)]
        pts = [(px(x), py(y)) for x, y in zip(xs_, ys)]
        d.line(pts, fill=col, width=4)
        if len(pts) <= 40:
            for p in pts:
                r = 6
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col, outline=BG, width=2)

    # x 轴刻度：最多 max_xticks 个，避免密集重叠
    uniq = sorted(set(all_x))
    if len(uniq) > max_xticks:
        idx = [round(i * (len(uniq) - 1) / (max_xticks - 1)) for i in range(max_xticks)]
        ticks = [uniq[i] for i in idx]
    else:
        ticks = uniq
    for x in ticks:
        d.text((px(x), mt + plot_h + 14), f"{x:g}", font=f_s, fill=MUTED, anchor="ma")

    d.text((ml + plot_w / 2, H - 62), xlabel, font=f_l, fill=INK, anchor="ma")
    d.text((ml - 60, mt), ylabel, font=f_l, fill=INK, anchor="lm")

    lx, ly = W - mr + 26, mt + 10
    for i, (name, _, _) in enumerate(series):
        col = SERIES[i % len(SERIES)]
        d.line([lx, ly, lx + 34, ly], fill=col, width=5)
        # 图例太长就换行，避免超出画布
        line = name
        while d.textlength(line, font=f_s) > mr - 40 and len(line) > 4:
            line = line[:-2]
        d.text((lx + 44, ly - 13), line, font=f_s, fill=INK)
        ly += 40

    img.save(out_path)


def bar_chart(
    out_path: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    ylabel: str,
    title: str,
    size: tuple[int, int] = (1500, 760),
    value_fmt: str = "{:.2f}",
) -> None:
    """分组柱状图。"""
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_t = _font(34)
    f_l = _font(24)
    f_s = _font(21)

    ml, mr, mt, mb = 120, 200, 90, 110
    plot_w, plot_h = W - ml - mr, H - mt - mb
    ymax, ystep = _nice_max(max(max(v) for _, v in series) * 1.15)

    def py(y):
        return mt + plot_h - y / ymax * plot_h

    d.text((ml, 28), title, font=f_t, fill=INK)

    steps = int(round(ymax / ystep))
    for i in range(steps + 1):
        y = ymax - i * ystep
        yy = py(y)
        d.line([ml, yy, ml + plot_w, yy], fill=GRID, width=1)
        d.text((ml - 16, yy - 12), f"{y:g}", font=f_s, fill=MUTED, anchor="ra")
    d.line([ml, mt + plot_h, ml + plot_w, mt + plot_h], fill=(170, 176, 188), width=2)

    n = len(categories)
    ns = len(series)
    group_w = plot_w / n
    bar_w = min(group_w * 0.72 / ns, 90)
    for i, cat in enumerate(categories):
        cx = ml + group_w * (i + 0.5)
        total_w = bar_w * ns
        for k, (name, vals) in enumerate(series):
            x0 = cx - total_w / 2 + k * bar_w
            y0 = py(vals[i])
            d.rectangle([x0, y0, x0 + bar_w - 8, mt + plot_h], fill=SERIES[k % len(SERIES)])
            d.text((x0 + (bar_w - 8) / 2, y0 - 10), value_fmt.format(vals[i]), font=f_s,
                   fill=INK, anchor="mb")
        d.text((cx, mt + plot_h + 14), cat, font=f_s, fill=MUTED, anchor="ma")

    d.text((ml - 60, mt), ylabel, font=f_l, fill=INK, anchor="lm")

    lx, ly = W - mr + 26, mt + 10
    for i, (name, _) in enumerate(series):
        d.rectangle([lx, ly - 10, lx + 26, ly + 10], fill=SERIES[i % len(SERIES)])
        d.text((lx + 38, ly - 13), name, font=f_s, fill=INK)
        ly += 40

    img.save(out_path)


def stacked_bar(
    out_path: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    ylabel: str,
    title: str,
    size: tuple[int, int] = (1500, 760),
) -> None:
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_t = _font(34)
    f_l = _font(24)
    f_s = _font(21)

    ml, mr, mt, mb = 120, 220, 90, 110
    plot_w, plot_h = W - ml - mr, H - mt - mb
    totals = [sum(s[1][i] for s in series) for i in range(len(categories))]
    ymax, ystep = _nice_max(max(totals) * 1.12)

    def py(y):
        return mt + plot_h - y / ymax * plot_h

    d.text((ml, 28), title, font=f_t, fill=INK)
    steps = int(round(ymax / ystep))
    for i in range(steps + 1):
        y = ymax - i * ystep
        d.line([ml, py(y), ml + plot_w, py(y)], fill=GRID, width=1)
        d.text((ml - 16, py(y) - 12), f"{y:g}", font=f_s, fill=MUTED, anchor="ra")
    d.line([ml, mt + plot_h, ml + plot_w, mt + plot_h], fill=(170, 176, 188), width=2)

    n = len(categories)
    group_w = plot_w / n
    bar_w = min(group_w * 0.42, 110)
    for i, cat in enumerate(categories):
        cx = ml + group_w * (i + 0.5)
        base = 0.0
        for k, (name, vals) in enumerate(series):
            y1 = py(base + vals[i])
            y0 = py(base)
            d.rectangle([cx - bar_w / 2, y1, cx + bar_w / 2, y0], fill=SERIES[k % len(SERIES)])
            base += vals[i]
        d.text((cx, py(base) - 10), f"{base:.0f}", font=f_s, fill=INK, anchor="mb")
        d.text((cx, mt + plot_h + 14), cat, font=f_s, fill=MUTED, anchor="ma")

    d.text((ml - 60, mt), ylabel, font=f_l, fill=INK, anchor="lm")
    lx, ly = W - mr + 26, mt + 10
    for i, (name, _) in enumerate(series):
        d.rectangle([lx, ly - 10, lx + 26, ly + 10], fill=SERIES[i % len(SERIES)])
        d.text((lx + 38, ly - 13), name, font=f_s, fill=INK)
        ly += 40
    img.save(out_path)
