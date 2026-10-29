# 图像拼接（全景图生成）—— 计算机视觉课程设计

输入两张有重叠的照片，自动估计几何变换、投影到同一画布并融合重叠区，输出一张无接缝的全景图。
核心几何计算（归一化 DLT、RANSAC/PROSAC、IRLS）全部手写实现，OpenCV 只用于特征提取与像素重采样。

![示例](outputs/figures/fig_04_panorama.png)

## 环境

- Python 3.10+（开发环境 3.12.14）
- Windows / Linux 均可，纯 CPU，无需 GPU
- 依赖：`numpy`、`opencv-python`、`pillow`

```bash
pip install -r requirements.txt
# 中国大陆网络可加：-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

拼接自己的两张图：

```bash
python main.py stitch --left 左图.jpg --right 右图.jpg --out results/
```

常用可选参数：

| 参数 | 说明 |
| --- | --- |
| `--detector ORB` | 换更快但精度略低的检测器 |
| `--blend feather` | 融合方式：naive / feather / multiband（默认） |
| `--ratio 0.8` | Lowe 比率阈值（默认 0.75） |
| `--threshold 3.0` | RANSAC 内点阈值（像素） |
| `--save-debug` | 额外保存匹配可视化与对位检查图 |

图形界面：

```bash
python app_gui.py
```

一键复现报告里的全部实验（约 2.5 分钟）：

```bash
python scripts/run_all.py      # 输出 outputs/figures/*.png 与 outputs/metrics.json
python scripts/make_report.py  # 生成 report/图像拼接课程报告.docx
```

## 目录结构

```
├── main.py                 # 命令行入口
├── app_gui.py              # tkinter 图形界面
├── src/
│   ├── config.py           # 全部可调参数
│   ├── data_gen.py         # 程序化场景 + 虚拟相机，构造带真值 H 的图像对
│   ├── features.py         # SIFT/ORB、KNN、Lowe 比率检验、双向一致性
│   ├── homography.py       # 归一化 DLT、RANSAC/PROSAC、IRLS、角点误差
│   ├── warp.py             # 画布计算、投影、最大内接矩形裁边
│   ├── blend.py            # 增益补偿、距离加权羽化、多频段融合
│   ├── metrics.py          # PSNR / SSIM / 接缝梯度
│   ├── pipeline.py         # 主流程 + 失败判定
│   ├── viz.py, charts.py   # 可视化与图表（中文标注）
│   └── io_utils.py         # 中文路径安全读写
└── scripts/                # 实验、报告生成脚本
```

## 效果

| 数据 | 匹配 / 内点 | 内点率 | 角点误差 | PSNR |
| --- | --- | --- | --- | --- |
| 合成街景（1700×1100） | 373 / 300 | 80.4% | 1.36 px | 37.82 dB |
| 真实照片平面场景 | 3063 / 3052 | 99.6% | 0.05 px | — |
| 手机实拍教室 | 30 / 12 | 40.0% | —（无真值） | — |

![实拍](outputs/figures/fig_06_real_shot.png)

## 注意

- 两张图需要有足够重叠（建议 ≥25%）。重叠过低时程序会给出失败原因而不是输出错位结果；
  但重叠率在 10%~20% 之间属于"能拼但不可靠"的灰区，请务必检查结果。
- 单应模型假设场景近似平面或相机只旋转。画面里有明显近景（桌子、人）时会出现局部重影，这是模型本身的限制。
- 处理大尺寸照片（>4000px）时建议先降采样，SIFT 耗时随像素数增长。
