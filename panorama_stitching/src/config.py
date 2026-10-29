from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """特征提取与匹配参数。"""

    detector: str = "SIFT"          # SIFT / ORB / AKAZE
    max_features: int = 0           # 0 表示不限制（SIFT 默认全提）
    contrast_threshold: float = 0.01  # SIFT 对比度阈值，越大点越少越稳
    use_clahe: bool = True          # 直方图均衡，缓解曝光差异
    ratio: float = 0.75             # Lowe 比率检验阈值
    cross_check: bool = True        # 是否做双向一致性校验
    nms_radius: float = 0.0         # 网格化筛选的关键点最小间距（像素）


@dataclass
class RansacConfig:
    """RANSAC 单应估计参数。"""

    mode: str = "custom"        # custom（手写 DLT+RANSAC） / opencv
    sampler: str = "prosac"     # prosac（按匹配质量优先采样） / uniform
    threshold: float = 3.0      # 内点阈值（像素），对称转移误差
    max_iters: int = 5000
    confidence: float = 0.999
    refine: bool = True         # 内点重估 + 加权迭代精化
    refine_iters: int = 5


@dataclass
class BlendConfig:
    """图像域融合参数。"""

    mode: str = "multiband"     # naive / feather / multiband
    levels: int = 5             # 拉普拉斯金字塔层数
    gain_compensation: bool = True   # 重叠区增益/偏置补偿
    exposure_model: str = "linear"   # gain / linear(ax+b)


@dataclass
class StitchConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)

    min_matches: int = 8        # 少于该匹配数直接判定失败
    min_inlier_ratio: float = 0.10
    max_reproj_error: float = 5.0
    output_max_width: int = 2600  # 全景图输出上限（仅影响保存尺寸）


DEFAULT = StitchConfig()
