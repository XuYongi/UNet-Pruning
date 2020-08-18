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
from PIL import Image, ImageEnhance
import numpy as np
import random



class RandomRotation:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            random_angle = np.random.randint(1, 360)
            image, label = image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)
        return  image, label


class RandomCropResize:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            image_width = image.size[0]
            image_height = image.size[1]
            size = (image_width, image_height)
            crop_win_size_w = np.random.randint(int(image_width * 0.8), image_width)
            crop_win_size_h = np.random.randint(int(image_height * 0.8), image_height)
            random_region = (
                (image_width - crop_win_size_w) >> 1, (image_height - crop_win_size_h) >> 1,
                (image_width + crop_win_size_w) >> 1,
                (image_height + crop_win_size_h) >> 1)

            image, label = image.crop(random_region), label.crop(random_region)

            image = image.resize(size, Image.BILINEAR)
            label = label.resize(size, Image.NEAREST)
        return image, label

class RandomColor:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            image, label = ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  # 调整图像锐度

        return image, label

class RandomGassion:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label):
        mean = 0.2
        sigma = 0.3
        if random.random() < self.prob:
            def gaussianNoisy(im, mean=0.2, sigma=0.3):
                """
                对图像做高斯噪音处理
                :param im: 单通道图像
                :param mean: 偏移量
                :param sigma: 标准差
                :return:
                """
                for _i in range(len(im)):
                    im[_i] += random.gauss(mean, sigma)
                return im

            assert image.size == label.size
            # 将图像转化成数组
            img = np.asarray(image)
            img.flags.writeable = True  # 将数组改为读写模式
            width, height = img.shape[:2]
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
            image, label = Image.fromarray(np.uint8(img)), label
        return image, label


randomRotation = RandomRotation(prob = 0.5)
randomCropResize = RandomCropResize(prob = 0.5)
randomColor = RandomColor(prob = 0.5)
randomGassion = RandomGassion(prob = 0.5)


def mytransform(image, mask):
    image, mask = randomRotation(image, mask)
    image, mask = randomCropResize(image, mask)
    image, mask = randomColor(image, mask)
    image, mask = randomGassion(image, mask)
    return image, mask


