# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset
import torch

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.hyp = hyp  # hyperparameters
        self.is_dual = (hyp.ch or 3) > 3
        self._modality_logged = False
        self.current_epoch = 0
        self.total_epochs = 300
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def set_epoch(self, epoch, total_epochs=None):
        """由 trainer 在每个 epoch 开始时调用，驱动渐进式训练阶段切换。"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs

    def _get_stage_ratios(self):
        """
        RG-PMMT 四阶段训练比例。
        返回 (p_dual, p_ir_only, p_rgb_only)。
          Stage 1 (0–50%T):   100/0/0    先学双模态互补
          Stage 2 (50–73%T):  80/10/10   轻度缺失适应
          Stage 3 (73–87%T):  60/20/20   增强缺失鲁棒性
          Stage 4 (87–100%T): 100/0/0    双模态回调微调
        """
        e = self.current_epoch
        T = max(self.total_epochs, 1)
        if e < round(0.500 * T):
            return 1.00, 0.00, 0.00
        elif e < round(0.733 * T):
            return 0.80, 0.10, 0.10
        elif e < round(0.867 * T):
            return 0.60, 0.20, 0.20
        else:
            return 1.00, 0.00, 0.00

    def _get_lambda_miss(self):
        """返回当前阶段的 lambda_miss 权重（用于 trainer 的损失缩放）。"""
        e = self.current_epoch
        T = max(self.total_epochs, 1)
        if e < round(0.500 * T) or e >= round(0.867 * T):
            return 0.0
        elif e < round(0.733 * T):
            return 0.2
        else:
            return 0.3

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        ir = f.replace("images", "imagesIR")          # ↳ 对应的红外文件
        # ir = f.replace("images", "imagesIR").replace("-1_", "-2_")
        # print(ir)
        if im is None:                             # not cached in RAM
            # ---------- ① 读取 ----------
            if fn.exists():                        # *.npy 缓存
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)             # BGR (RGB)
            else:
                im = cv2.imread(f)                 # BGR
                if self.is_dual:                   # ← 是否需要 IR
                    ir_im = cv2.imread(ir)
                    if ir_im is None:
                        raise FileNotFoundError(f"IR image not found: {ir}")
                    im = cv2.merge((ir_im, im))    # IR + RGB → 6 通道
                    # im = np.concatenate((ir, ir_im), axis=2)

            # ---------- ② 首次打印 ----------
            if not self._modality_logged:
                LOGGER.info(f"{self.prefix}Loaded "
                            f"{'✅dual‑modality (RGB+IR)' if self.is_dual else 'single‑modality (RGB only)'} dataset")
                self._modality_logged = True

            if im is None:
                raise FileNotFoundError(f"Image not found: {f}")

            # ---------- ③ 尺寸变换 ----------
            h0, w0 = im.shape[:2]                  # 原始高宽
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # ---------- ④ 缓存到内存 buffer ----------
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j] = self.im_hw0[j] = self.im_hw[j] = None

            return im, (h0, w0), im.shape[:2]

        # cached branch
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    # import random
    # import numpy as np
    #
    # def apply_modality_dropout(self, img):
    #     """
    #     Feature Modality Dropout Strategy (Grayscale Substitution)
    #
    #     采用完全随机策略：
    #     33.3% dual modality
    #     33.3% top light only
    #     33.3% bottom light only
    #     """
    #
    #     if not self.is_dual or not self.augment:
    #         return img, "dual"
    #
    #     # 直接从三种状态中等概率随机抽取一个
    #     states = ["dual", "top_only", "bottom_only"]
    #     modality = random.choice(states)
    #
    #     if modality == "dual":
    #         # 双模态正常输入，直接返回
    #         pass
    #
    #     elif modality == "top_only":
    #         # 只有上打光 (保留 RGB)，下打光 (IR) 替换为 RGB 的灰度图
    #         rgb_img = img[..., 3:6].astype(np.float32)
    #         gray_rgb = 0.299 * rgb_img[..., 0] + 0.587 * rgb_img[..., 1] + 0.114 * rgb_img[..., 2]
    #         gray_rgb_3c = np.stack([gray_rgb, gray_rgb, gray_rgb], axis=-1).astype(img.dtype)
    #         img[..., 0:3] = gray_rgb_3c
    #
    #     else:
    #         # 只有下打光 (保留 IR)，上打光 (RGB) 替换为 IR 的灰度图
    #         ir_img = img[..., 0:3].astype(np.float32)
    #         gray_ir = 0.299 * ir_img[..., 0] + 0.587 * ir_img[..., 1] + 0.114 * ir_img[..., 2]
    #         gray_ir_3c = np.stack([gray_ir, gray_ir, gray_ir], axis=-1).astype(img.dtype)
    #         img[..., 3:6] = gray_ir_3c
    #
    #     return img, modality
###################################老师想的方法####################
    def apply_modality_dropout(self, img):
        """
        RG-PMMT: 均值填充 + 模态标志，渐进式混合模态训练。

        通道定义:
        [0:3] -> 下打光 (IR/反射光)
        [3:6] -> 上打光 (RGB/透射光)

        缺失模态用均值常量 114 填充（不再用另一模态灰度图伪造），
        网络通过 channel_valid 标志知道哪路不可用。
        训练比例由 _get_stage_ratios() 按当前 epoch 动态决定。
        """
        if not self.is_dual or not self.augment:
            return img, "dual"

        p_dual, p_ir_only, _ = self._get_stage_ratios()
        r = random.random()

        if r < p_dual:
            modality = "dual"
        elif r < p_dual + p_ir_only:
            # 仅下打光 (IR) 可用，上打光 (RGB) 缺失 → 均值填充 RGB 通道
            img[..., 3:6] = 114
            modality = "bottom_only"
        else:
            # 仅上打光 (RGB) 可用，下打光 (IR) 缺失 → 均值填充 IR 通道
            img[..., 0:3] = 114
            modality = "top_only"

        return img, modality
    # def apply_modality_dropout(self, img, epoch=0):
    #     """
    #     重构后的混合模态策略：均值填充 + 动态阶段控制
    #     [0:3] -> 反射光 (RGB/Top), [3:6] -> 透射光 (IR/Bottom)
    #     """
    #     if not self.is_dual or not self.augment:
    #         return img, torch.tensor([1.0, 1.0])
    #
    #     # 1. 动态获取当前阶段的比例 (根据 300 epoch 设计)
    #     if epoch < 150:
    #         p_bi, p_top, p_bottom = 1.0, 0.0, 0.0  # Stage 1
    #     elif epoch < 220:
    #         p_bi, p_top, p_bottom = 0.8, 0.1, 0.1  # Stage 2
    #     elif epoch < 260:
    #         p_bi, p_top, p_bottom = 0.6, 0.2, 0.2  # Stage 3
    #     else:
    #         p_bi, p_top, p_bottom = 1.0, 0.0, 0.0  # Stage 4: 回调
    #
    #     p = random.random()
    #     m_r, m_t = 1.0, 1.0  # 默认双模态有效
    #
    #     # 2. 执行均值填充 (Mean Filling)
    #     if p < p_top:
    #         # 仅保留上打光 (反射)，下打光 (透射) 填充均值
    #         m_t = 0.0
    #         # 计算透射光分支的均值（或者直接用训练集全局均值）
    #         avg_val = np.mean(img[..., 0:3], axis=(0, 1))
    #         img[..., 0:3] = avg_val
    #         modality = "top_only"
    #     elif p < (p_top + p_bottom):
    #         # 仅保留下打光 (透射)，上打光 (反射) 填充均值
    #         m_r = 0.0
    #         avg_val = np.mean(img[..., 3:6], axis=(0, 1))
    #         img[..., 3:6] = avg_val
    #         modality = "bottom_only"
    #     else:
    #         modality = "dual"
    #
    #     mask = torch.tensor([m_r, m_t])
    #     return img, mask
###################################常用############################
    # def apply_modality_dropout(self, img):
    #     """
    #         Feature Modality Dropout Strategy (Grayscale Substitution)
    #
    #         通道定义:
    #         [0:3] -> 下打光 (IR)
    #         [3:6] -> 上打光 (RGB)
    #
    #         训练比例:
    #         90% dual modality
    #         5% top light only (IR 替换为 RGB 的灰度图)
    #         5% bottom light only (RGB 替换为 IR 的灰度图)
    #         """
    #
    #     if not self.is_dual or not self.augment:
    #         return img, "dual"
    #
    #     p = random.random()
    #
    #     if p < 0.0:
    #         # 双模态正常输入
    #         modality = "dual"
    #
    #     elif p < 0.5:
    #         # 只有上打光 (保留 RGB)，下打光 (IR) 替换为 RGB 的灰度图
    #         rgb_img = img[..., 3:6].astype(np.float32)
    #         gray_rgb = 0.299 * rgb_img[..., 0] + 0.587 * rgb_img[..., 1] + 0.114 * rgb_img[..., 2]
    #         gray_rgb_3c = np.stack([gray_rgb, gray_rgb, gray_rgb], axis=-1).astype(img.dtype)
    #         img[..., 0:3] = gray_rgb_3c
    #         modality = "top_only"
    #
    #     else:
    #         # 只有下打光 (保留 IR)，上打光 (RGB) 替换为 IR 的灰度图
    #         ir_img = img[..., 0:3].astype(np.float32)
    #         gray_ir = 0.299 * ir_img[..., 0] + 0.587 * ir_img[..., 1] + 0.114 * ir_img[..., 2]
    #         gray_ir_3c = np.stack([gray_ir, gray_ir, gray_ir], axis=-1).astype(img.dtype)
    #         img[..., 3:6] = gray_ir_3c
    #         modality = "bottom_only"
    #
    #     return img, modality


    # def apply_modality_dropout(self, img):
    #     """
    #     Feature Modality Dropout Strategy (Constant Value Substitution)
    #
    #     通道定义:
    #     [0:3] -> 下打光 (IR)
    #     [3:6] -> 上打光 (RGB)
    #     """
    #
    #     if not self.is_dual or not self.augment:
    #         return img, "dual"
    #
    #     p = random.random()
    #
    #     if p < 0.9:
    #         # 双模态正常输入
    #         modality = "dual"
    #
    #     elif p < 0.95:
    #         # 丢弃下打光 (IR)，用 128 像素值填充
    #         # img[..., 0:3].shape 会自动获取高、宽和通道数
    #         img[..., 0:3] = 128
    #         modality = "top_only"
    #
    #     else:
    #         # 丢弃上打光 (RGB)，用 128 像素值填充
    #         img[..., 3:6] = 128
    #         modality = "bottom_only"
    #
    #     return img, modality
    # def apply_modality_dropout(self, img):
    #     """
    #     Feature Modality Dropout Strategy
    #
    #     通道定义:
    #     [0:3] -> 下打光 (IR)
    #     [3:6] -> 上打光 (RGB)
    #
    #     训练比例:
    #     70% dual modality
    #     15% top light only
    #     15% bottom light only
    #     """
    #
    #     if not self.is_dual or not self.augment:
    #         return img, "dual"
    #
    #     p = random.random()
    #
    #     if p < 0.9:
    #         # 双模态正常输入
    #         modality = "dual"
    #
    #     elif p < 0.95:
    #         # 只有上打光 (RGB)
    #         # IR 模态置零
    #         img[..., 0:3] = 0
    #         modality = "top_only"
    #
    #     else:
    #         # 只有下打光 (IR)
    #         # RGB 模态置零
    #         img[..., 3:6] = 0
    #         modality = "bottom_only"
    #
    #     return img, modality
    # def apply_modality_dropout(self, img):
    #     """
    #     Modality Dropout with REPLICATION strategy.
    #     确保物理意义、代码索引和注释三者对齐。
    #     [0:3] -> 下打光 (IR), [3:6] -> 上打光 (RGB)
    #     """
    #     if not self.is_dual or not self.augment:
    #         return img, "dual"
    #
    #     p = random.random()
    #
    #     if p < 0.7:
    #         # 双灯全亮模式
    #         modality = "dual"
    #     elif p < 0.85:
    #         # 【修正】：模拟“只有上打光(RGB)”
    #         # 做法：把上打光(3:6)的内容复制到下打光支路(0:3)
    #         # 结果：[RGB, RGB]
    #         img[..., 0:3] = img[..., 3:6].copy()
    #         modality = "top_only"
    #     else:
    #         # 【修正】：模拟“只有下打光(IR)”
    #         # 做法：把下打光(0:3)的内容复制到上打光支路(3:6)
    #         # 结果：[IR, IR]
    #         img[..., 3:6] = img[..., 0:3].copy()
    #         modality = "bottom_only"
    #
    #     return img, modality



    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """Check image caching requirements vs available disk space."""
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = cv2.imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    # def __getitem__(self, index):
    #     """Returns transformed label information for given index."""
    #     return self.transforms(self.get_image_and_label(index))
    def __getitem__(self, index):
        """Returns transformed label information with modality protection."""
        # 1. 获取原始数据（包含我们注入的 modality）
        label = self.get_image_and_label(index)
        
        # 2. 暂存这些关键元数据，防止被 transforms 删掉
        modality = label.get("modality", "dual")
        # 顺便根据 modality 生成 channel_valid 标志 (1为有效, 0为置零)
        if modality == "top_only":
            channel_valid = [0, 0, 0, 1, 1, 1]
        elif modality == "bottom_only":
            channel_valid = [1, 1, 1, 0, 0, 0]
        else:
            channel_valid = [1, 1, 1, 1, 1, 1]

        # 3. 执行数据增强 (Mosaic, Mixup, Format 等会在这里运行)
        label = self.transforms(label)
        
        # 4. ⚠️ 核心修复：手动将丢失的字段重新注入增强后的字典
        label["modality"] = modality
        label["channel_valid"] = torch.tensor(channel_valid)
        
        return label

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["img"], label["modality"] = self.apply_modality_dropout(label["img"])
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
