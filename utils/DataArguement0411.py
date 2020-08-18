# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

# 导入数据增强工具
import Augmentor

# 确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline("../../data/train/test/imgs/")
p.ground_truth("../../data/train/test/masks/")

# 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
p.rotate(probability=0.8, max_left_rotation=10, max_right_rotation=10)
p.random_distortion(probability=0.3, grid_width=5, grid_height=5, magnitude=10)  # 小块变形
# 图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.5)
p.skew_tilt(probability=0.2,magnitude=0.1)
# 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
p.zoom_random(probability=0.2, percentage_area=0.9)
p.greyscale(0.3)
p.random_color(0.4,0.2,1)
# 最终扩充的数据样本数
p.sample(200)
