from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import BlendConfig, FeatureConfig, RansacConfig, StitchConfig
from src.io_utils import imread, imwrite
from src.pipeline import stitch

PREVIEW_W, PREVIEW_H = 760, 380


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("图像拼接 · 全景图生成")
        root.geometry("1020x760")
        try:
            root.iconbitmap(default="")
        except Exception:
            pass

        self.left_path = tk.StringVar()
        self.right_path = tk.StringVar()
        self.detector = tk.StringVar(value="SIFT")
        self.blend = tk.StringVar(value="multiband")
        self.ratio = tk.DoubleVar(value=0.75)
        self.status = tk.StringVar(value="请选择左右两张有重叠的照片")
        self.result = None
        self._photo = None

        self._build()

    def _build(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        for i, (label, var) in enumerate([("左图", self.left_path), ("右图", self.right_path)]):
            ttk.Label(top, text=label, width=6).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var, width=70).grid(row=i, column=1, sticky="we", padx=6)
            ttk.Button(top, text="浏览…", command=lambda v=var: self._pick(v)).grid(row=i, column=2)
        top.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(self.root, text="参数", padding=8)
        opts.pack(fill="x", padx=10, pady=6)

        ttk.Label(opts, text="特征检测器").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.detector, values=["SIFT", "ORB"], width=8,
                     state="readonly").grid(row=0, column=1, padx=(4, 18))

        ttk.Label(opts, text="融合方式").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self.blend, values=["multiband", "feather", "naive"],
                     width=12, state="readonly").grid(row=0, column=3, padx=(4, 18))

        ttk.Label(opts, text="Lowe 比率").grid(row=0, column=4, sticky="w")
        ttk.Scale(opts, from_=0.60, to=0.90, variable=self.ratio, orient="horizontal",
                  length=140).grid(row=0, column=5, padx=4)
        ttk.Label(opts, textvariable=self.ratio).grid(row=0, column=6)

        btns = ttk.Frame(self.root, padding=(10, 0))
        btns.pack(fill="x")
        self.btn_run = ttk.Button(btns, text="开始拼接", command=self._run)
        self.btn_run.pack(side="left")
        ttk.Button(btns, text="保存全景图", command=self._save).pack(side="left", padx=8)
        ttk.Label(btns, textvariable=self.status, foreground="#2b6cb0").pack(side="left", padx=12)

        self.canvas = tk.Canvas(self.root, bg="#f4f5f7", highlightthickness=1,
                                highlightbackground="#d8dbe0")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=8)

    def _pick(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("全部", "*.*")],
        )
        if path:
            var.set(path)

    def _run(self):
        if not self.left_path.get() or not self.right_path.get():
            messagebox.showwarning("提示", "请先选择左右两张图片")
            return
        self.btn_run.config(state="disabled")
        self.status.set("正在拼接…")
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        try:
            left = imread(self.left_path.get())
            right = imread(self.right_path.get())
            if left is None or right is None:
                raise ValueError("图片读取失败（路径可能包含无法识别的字符）")
            cfg = StitchConfig(
                features=FeatureConfig(detector=self.detector.get(), ratio=float(self.ratio.get())),
                ransac=RansacConfig(),
                blend=BlendConfig(mode=self.blend.get()),
            )
            res = stitch(left, right, cfg)
            if not res.ok:
                self.root.after(0, lambda: self._fail(res.message))
                return
            self.result = res
            q = res.quality
            msg = (f"完成：匹配 {q.get('n_matches')} 对 / 内点 {q.get('n_inliers')} 个，"
                   f"全景 {res.pano.shape[1]}×{res.pano.shape[0]}，"
                   f"耗时 {res.timings.get('total', 0):.2f}s")
            self.root.after(0, lambda: self._show(res.pano, msg))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda e=exc: self._fail(str(e)))

    def _fail(self, msg: str):
        self.status.set("失败：" + msg)
        self.btn_run.config(state="normal")
        messagebox.showerror("拼接失败", msg)

    def _show(self, pano: np.ndarray, msg: str):
        h, w = pano.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h, 1.0)
        small = cv2.resize(pano, (max(1, int(w * scale)), max(1, int(h * scale))))
        img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                 image=self._photo)
        self.status.set(msg)
        self.btn_run.config(state="normal")

    def _save(self):
        if self.result is None:
            messagebox.showinfo("提示", "还没有可以保存的结果")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        if imwrite(path, self.result.pano):
            messagebox.showinfo("已保存", path)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
