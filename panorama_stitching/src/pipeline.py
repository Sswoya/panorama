from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import blend as blend_mod
from . import features as feat_mod
from . import homography as hom_mod
from . import metrics as met_mod
from .config import StitchConfig
from .warp import compute_canvas, crop_rect, largest_inscribed_rect, warp_pair


@dataclass
class StitchResult:
    ok: bool
    pano: np.ndarray | None = None
    H: np.ndarray | None = None
    quality: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    message: str = ""
    warped: dict | None = None
    match: object | None = None
    hres: object | None = None
    debug: dict = field(default_factory=dict)


def _homography_is_sane(H: np.ndarray, w: int, h: int) -> tuple[bool, str]:
    if H is None or not np.all(np.isfinite(H)):
        return False, "单应矩阵含非法值"
    if abs(H[2, 2]) < 1e-8:
        return False, "单应矩阵退化"
    Hn = H / H[2, 2]
    det = np.linalg.det(Hn[:2, :2])
    if det <= 0:
        return False, "单应矩阵带镜像/翻转（行列式<=0）"
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    p = hom_mod.apply_h(Hn, corners)
    span = np.linalg.norm(p.max(axis=0) - p.min(axis=0))
    if span > 6.0 * max(w, h):
        return False, "投影范围异常（疑似错误匹配）"
    if span < 0.15 * max(w, h):
        return False, "投影范围过小（疑似整体缩放错误）"
    return True, "ok"


def gt_panorama(pair, T: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    if pair is None or pair.scene is None or pair.A_right is None:
        return None
    M = T @ pair.A_right
    scene = pair.scene
    gain = float(pair.meta.get("gain", 1.0))
    cast = pair.meta.get("color_cast", [1.0, 1.0, 1.0])
    sim = scene.astype(np.float32) * gain * np.array(cast, dtype=np.float32)[None, None, :]
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return cv2.warpPerspective(sim, M, size, flags=cv2.INTER_LINEAR)


def stitch(
    left: np.ndarray,
    right: np.ndarray,
    cfg: StitchConfig | None = None,
    pair=None,
    verbose: bool = False,
) -> StitchResult:
    cfg = cfg or StitchConfig()
    timings: dict[str, float] = {}
    t_all = time.perf_counter()
    hl, wl = left.shape[:2]

    #特征检测 + 匹配
    m = feat_mod.match(left, right, cfg.features)
    timings["features"] = time.perf_counter() - t_all
    if verbose:
        print(f"  [1/5] 特征: {len(m.kp1)} / {len(m.kp2)} 点，匹配 {len(m)} 对，{timings['features']*1000:.0f} ms")

    if len(m) < cfg.min_matches:
        return StitchResult(
            ok=False,
            message=f"匹配点过少（{len(m)} < {cfg.min_matches}），拒绝拼接",
            timings=timings,
            match=m,
        )

    #RANSAC 求单应
    t = time.perf_counter()
    hres = hom_mod.estimate(m.pts1, m.pts2, cfg.ransac)
    timings["ransac"] = time.perf_counter() - t
    if verbose:
        print(
            f"  [2/5] RANSAC: 内点 {hres.n_inliers}/{len(m)} "
            f"({hres.inlier_ratio*100:.1f}%)，迭代 {hres.n_iters} 次，"
            f"RMS {hres.rms_error:.2f}px，{timings['ransac']*1000:.0f} ms"
        )

    if hres.n_inliers < cfg.min_matches:
        return StitchResult(False, message=f"内点不足（{hres.n_inliers}），判定拼接失败", timings=timings, match=m, hres=hres)
    if hres.inlier_ratio < cfg.min_inlier_ratio:
        return StitchResult(
            False,
            message=f"内点率过低（{hres.inlier_ratio*100:.1f}%），判定拼接失败",
            timings=timings,
            match=m,
            hres=hres,
        )

    ok, why = _homography_is_sane(hres.H, wl, hl)
    if not ok:
        return StitchResult(False, message=f"单应矩阵不合理：{why}", timings=timings, match=m, hres=hres)

    #投影到全景画布
    t = time.perf_counter()
    warped = warp_pair(left, right, hres.H)
    timings["warp"] = time.perf_counter() - t

    #融合
    t = time.perf_counter()
    pano, binfo = blend_mod.blend_pair(warped, cfg.blend)
    timings["blend"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_all

    #指标
    ml, mr = warped["mask_left"], warped["mask_right"]
    quality: dict = {
        "n_kp_left": len(m.kp1),
        "n_kp_right": len(m.kp2),
        "n_matches": len(m),
        "n_inliers": hres.n_inliers,
        "inlier_ratio": hres.inlier_ratio,
        "ransac_rms": hres.rms_error,
        "coverage": met_mod.coverage(pano, ml | mr),
        "overlap_ratio": float((ml & mr).sum() / max(min(ml.sum(), mr.sum()), 1)),
        "seam_energy": met_mod.seam_energy(pano, ml, mr),
        "gain": binfo.get("gain", {}),
        "canvas": warped["size"],
    }

    if pair is not None:
        quality["corner_rmse_gt"] = hom_mod.corner_error(hres.H, pair.H_gt, wl, hl)
        gt = gt_panorama(pair, warped["T"], warped["size"])
        if gt is not None:
            valid = cv2.erode((ml | mr).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            quality["psnr_gt"] = met_mod.psnr(pano, gt, valid)
            quality["ssim_gt"] = met_mod.ssim(pano, gt, valid)

    # 裁掉投影产生的黑色楔形边，输出一个规整矩形
    rect = largest_inscribed_rect(ml | mr)
    pano_rect = crop_rect(pano, rect)
    gt_full = gt_panorama(pair, warped["T"], warped["size"]) if pair is not None else None
    quality["crop_area_ratio"] = float((rect[2] * rect[3]) / max(pano.shape[0] * pano.shape[1], 1))

    return StitchResult(
        ok=True,
        pano=pano_rect,
        H=hres.H,
        quality=quality,
        timings=timings,
        message="拼接成功",
        warped=warped,
        match=m,
        hres=hres,
        debug={
            "pano_full": pano,
            "pano_rect": rect,
            "gt_pano": gt_full,
            "gt_pano_crop": crop_rect(gt_full, rect) if gt_full is not None else None,
        },
    )
