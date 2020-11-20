import random
from abc import ABC
from scipy import stats
import numpy as np
import math
import cv2
from Project.mnist import *


class Compose:
    """
    组合使用上面的方法，模仿tf.Compose()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class OneOf:
    """
    随机选择策略列表中的一种策略增强图像
    建议把所有加噪声的方法加入此类
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        n = len(self.transforms)
        idx = random.randint(0, n - 1)
        t = self.transforms[idx]
        return t(img)


class Strategy(ABC):
    def __init__(self):
        pass

    def __call__(self, img):
        pass


class Wrapping(Strategy):
    """
    图像扭曲
    """

    def __init__(self, alpha=1.2):
        super(Wrapping, self).__init__()
        self.alpha = alpha

    def __call__(self, img):
        h, w = img.shape[:2]
        img_out = img.copy()
        for r in range(h):
            for c in range(w):
                offset_x = int(self.alpha *
                               (stats.norm.cdf(2 * self.alpha * r / h - self.alpha) - 0.5))
                offset_y = int(self.alpha *
                               (stats.norm.cdf(2 * self.alpha * c / w - self.alpha) - 0.5))
                if r + offset_x < h and c + offset_y < w:
                    img_out[r, c] = img[r + offset_x, c + offset_y]
                else:
                    img_out[r, c] = 255.
        return img_out


class RandomRotate(Strategy):
    def __init__(self, angle, zoom):
        super(RandomRotate, self).__init__()
        self.angel = np.random.uniform(-angle, angle)
        self.zoom = np.random.uniform(zoom[0], zoom[1])

    def __call__(self, img):
        h, w = img.shape[:2]
        center_coordinate = (int(w / 2), int(h / 2))
        M = cv2.getRotationMatrix2D(center_coordinate, self.angel, self.zoom)
        rotated_img = cv2.warpAffine(img, M, (h, w), borderMode=cv2.BORDER_REPLICATE)
        return rotated_img


class GaussNoise(Strategy):
    def __init__(self, mean=(0, 64), std=(0.01, 0.1)):
        super(GaussNoise, self).__init__()
        self.mean = np.random.uniform(mean[0], mean[1])
        self.std = np.random.uniform(std[0], std[1])

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, img.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = img + noise  # 将噪声和原始图像进行相加得到加噪后的图像
        out = np.clip(out, 0, 255).astype(dtype=np.uint8)  # clip函数将元素的大小限制在了0和1之间了，小于的用low_clip代替，大于1的用1代替
        return out


class SaltAndPepper(Strategy):
    def __init__(self, p=0.025):
        super(SaltAndPepper, self).__init__()
        self.p = p

    def __call__(self, img):
        shape = img.shape
        img = img.flatten()
        output = img.copy()
        threshold = 1 - self.p
        for i in range(img.shape[0]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < self.p:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                output[i] = 0
            elif rdn > threshold:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                output[i] = 255
        return output.reshape(shape)


class GaussBlur(Strategy):
    """实现参考：http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.htmlzz"""

    def __init__(self, sigma, radius):
        super(GaussBlur, self).__init__()
        self.sigma = sigma
        self.radius = radius

    # 高斯函数
    def _gauss_func(self, x, y):
        m = 1 / (2 * math.pi * self.sigma * self.sigma)
        n = math.exp(-(x * x + y * y) / (2 * self.sigma * self.sigma))
        return m * n

    # 高斯核
    def _get_kernel(self):
        n = self.radius * 2 + 1
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                result[i, j] = self._gauss_func(i - self.radius, j - self.radius)
        tot = result.sum()
        return result / tot

    def _filter(self, img, kernel):
        h, w = img.shape[:2]
        out = img.copy()
        for i in range(self.radius, h - self.radius):
            for j in range(self.radius, w - self.radius):
                t = img[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                a = np.multiply(t, kernel)
                out[i, j] = a.sum()
        return out.astype(dtype=np.uint8)

    def __call__(self, img):
        kernel = self._get_kernel()
        out = self._filter(img, kernel)
        return out


class RandomSharpen(Strategy):
    def __init__(self, slope, bias):
        super(RandomSharpen, self).__init__()
        self.slope = np.random.uniform(slope[0], slope[1])
        self.bias = np.random.uniform(bias[0], bias[1])

    def __call__(self, img):
        avg = np.mean(img)
        # 图像亮度和对比度调整
        res = (img - avg) * self.slope + avg + self.bias
        res = np.clip(res, 0, 255)
        return res.astype(dtype=np.uint8)


# class ReverseColor(Strategy):
#     def __init__(self):
#         super(ReverseColor, self).__init__()
#
#     def __call__(self, img):
#         out = 255 - img
#         return out


geoTrans = Compose([
    Wrapping(),
    RandomRotate(20, (0.7, 1.2)),
])

noiser = OneOf([
    GaussNoise(),
    SaltAndPepper(),
])

sharpOrBlur = OneOf([
    GaussBlur(0.1, 1),
    RandomSharpen(slope=(1.2, 1.5), bias=(0, 64)),
])

strategy = [geoTrans, noiser, sharpOrBlur]  # 策略表，挨个尝试，比较扩增效果

# funcs = [GaussBlur(0.05, 1), RandomSharpen(slope=(1.2, 1.5), bias=(0, 64))]
# aug_images = np.array([])
# aug_labels
