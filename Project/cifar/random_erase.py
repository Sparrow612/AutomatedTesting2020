"""
@author: CHENG Rongxin
@email: 181250021@smail.nju.edu.cn
"""
import random
import math
import torch


class RandomErasing:
    """
    我自己实现的随机擦除算法，算法来自2020年AAAI的论文'Random Erasing Data Augmentation'
    初始化函数中的默认参数是论文中实验结果最好的参数
    设计方法参考了torchvision.transforms，按照我自己的实际需求进行了简化
    """

    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        """
        :param scale: 擦除区域面积:图像面积范围
        :param ratio: 擦除区域的H:W范围
        """
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        return self._process(img)

    def _process(self, img):
        o_c, o_h, o_w = img.shape
        area = o_h * o_w
        h, w = o_h, o_w
        while not (h < o_h and w < o_w):
            erase_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            # 由擦除面积和高宽比可以计算出擦除区域高和宽，原理可见论文原文，或见本项目最终报告
            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
        i = random.randint(0, o_h - h)
        j = random.randint(0, o_w - w)
        coverage = torch.empty(o_c, h, w, dtype=torch.float32).normal_()
        # 随机生成覆盖区域，并对它标准正则化
        img = img.clone()  # 防止影响到源图像
        img[:, i:i + h, j:j + w] = coverage
        return img
