"""命令行入口：拼接任意两张有重叠的照片。

用法：
    python main.py stitch --left 左图.jpg --right 右图.jpg --out results/
    python main.py stitch --left a.jpg --right b.jpg --blend feather --detector ORB
    python main.py demo            # 用程序化场景自检（需要真值时可看指标）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BlendConfig, FeatureConfig, RansacConfig, StitchConfig
from src.io_utils import imread, imwrite
from src.pipeline import stitch


def build_cfg(args) -> StitchConfig:
    return StitchConfig(
        features=FeatureConfig(
            detector=args.detector,
            ratio=args.ratio,
            cross_check=not args.no_cross_check,
            use_clahe=not args.no_clahe,
            max_features=args.max_features,
        ),
        ransac=RansacConfig(
            threshold=args.threshold,
            sampler=args.sampler,
            mode="opencv" if args.use_opencv else "custom",
        ),
        blend=BlendConfig(
            mode=args.blend,
            levels=args.levels,
            gain_compensation=not args.no_gain,
        ),
    )


def cmd_stitch(args) -> int:
    left = imread(args.left)
    right = imread(args.right)
    if left is None or right is None:
        print(f"[错误] 读不到图片：{args.left if left is None else args.right}")
        return 2

    print(f"左图 {left.shape[1]}×{left.shape[0]}   右图 {right.shape[1]}×{right.shape[0]}")
    cfg = build_cfg(args)
    t0 = time.perf_counter()
    res = stitch(left, right, cfg, None, verbose=True)
    elapsed = time.perf_counter() - t0

    if not res.ok:
        print(f"[失败] {res.message}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    stem = args.name or "panorama"
    pano_path = os.path.join(args.out, f"{stem}.png")
    imwrite(pano_path, res.pano)
    print(f"[完成] 全景图 {res.pano.shape[1]}×{res.pano.shape[0]} -> {pano_path}  "
          f"({elapsed:.2f}s)")

    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.quality.items()}
    metrics["H"] = res.H.tolist() if res.H is not None else None
    metrics["timings"] = {k: round(v, 4) for k, v in res.timings.items()}
    with open(os.path.join(args.out, f"{stem}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("       指标 ->", os.path.join(args.out, f"{stem}_metrics.json"))

    if args.save_debug:
        from src import viz

        fig = viz.fig_matches(left, right, res.match, res.hres)
        imwrite(os.path.join(args.out, f"{stem}_matches.png"), fig)
        imwrite(os.path.join(args.out, f"{stem}_overlay.png"), viz.fig_overlay(res.warped))
    print("H =\n", res.H)
    return 0


def cmd_demo(args) -> int:
    from src.data_gen import Camera, make_pair, render_street

    print("生成程序化场景（约 4 秒）…")
    pair = make_pair(render_street, Camera(), "demo", gain=0.8, noise_sigma=2.0)
    res = stitch(pair.left, pair.right, StitchConfig(), pair, verbose=True)
    if not res.ok:
        print("[失败]", res.message)
        return 1
    os.makedirs(args.out, exist_ok=True)
    imwrite(os.path.join(args.out, "demo_left.png"), pair.left)
    imwrite(os.path.join(args.out, "demo_right.png"), pair.right)
    imwrite(os.path.join(args.out, "demo_panorama.png"), res.pano)
    print(f"[完成] PSNR {res.quality.get('psnr_gt', 0):.2f} dB, "
          f"SSIM {res.quality.get('ssim_gt', 0):.4f}, "
          f"角点误差 {res.quality.get('corner_rmse_gt', 0):.2f} px")
    print("       输出目录 ->", args.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="图像拼接 / 全景图生成")
    sub = ap.add_subparsers(dest="cmd", required=False)

    s = sub.add_parser("stitch", help="拼接自己的两张图片")
    s.add_argument("--left", required=True)
    s.add_argument("--right", required=True)
    s.add_argument("--out", default="results")
    s.add_argument("--name", default=None, help="输出文件名前缀")
    s.add_argument("--detector", default="SIFT", choices=["SIFT", "ORB"])
    s.add_argument("--blend", default="multiband", choices=["naive", "feather", "multiband"])
    s.add_argument("--ratio", type=float, default=0.75)
    s.add_argument("--threshold", type=float, default=3.0)
    s.add_argument("--levels", type=int, default=5)
    s.add_argument("--sampler", default="prosac", choices=["prosac", "uniform"])
    s.add_argument("--max-features", type=int, default=0)
    s.add_argument("--no-gain", action="store_true", help="关闭曝光增益补偿")
    s.add_argument("--no-clahe", action="store_true")
    s.add_argument("--no-cross-check", action="store_true")
    s.add_argument("--use-opencv", action="store_true", help="用 cv2.findHomography 代替手写 RANSAC")
    s.add_argument("--save-debug", action="store_true", help="额外保存匹配图与对位图")
    s.set_defaults(func=cmd_stitch)

    d = sub.add_parser("demo", help="用程序化场景自检")
    d.add_argument("--out", default="results")
    d.set_defaults(func=cmd_demo)

    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
