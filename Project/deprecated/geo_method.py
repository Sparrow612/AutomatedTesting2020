import random
import cv2
import os
from collections import defaultdict
import numpy as np


class GeoAugmenter:
    """
    因为人为确定几何变化程度太过主观，所以这里所有的几何变化方法都使用随机方法
    一共三个扩增方法：
        1. 随机旋转
        2. 随机平移
        3. 随机放缩
    """

    def __init__(self, imgs, lbls):
        self.images, self.labels = imgs, lbls
        self.results = defaultdict(list)

    def random_rotate(self, limit=15, zoom=0.3):
        """
            仿射变化
            :param zoom: 旋转后放缩范围
            :param limit:旋转角度上下限
            :return: 旋转后的图像
            """
        # 旋转矩阵
        for img in self.images:
            img = img.reshape(28, 28)
            rows, cols = img.shape[:2]
            center_coordinate = (int(cols / 2), int(rows / 2))
            angle = random.uniform(-limit, limit)
            zoom_rate = random.uniform(1 - zoom, 1 + zoom)
            M = cv2.getRotationMatrix2D(center_coordinate, angle, zoom_rate)
            # 仿射变换
            rotate_img = cv2.warpAffine(img, M, (rows, cols), borderMode=cv2.BORDER_REPLICATE)
            self.results['random_rotate'].append(rotate_img.flatten())

    def random_shift(self, limit=5):
        """
        随机平移图像
        :param limit: 平移长度限制
        :return: 平移后的图片，第一个为左移，第二个为右移
        """
        for img in self.images:
            img = img.reshape(28, 28)
            ls_image, rs_image = np.array([]), np.array([])
            s_len = random.randint(1, limit)
            for row in img:
                rs_image = np.append(rs_image,
                                     (np.append(row[-s_len:], row[:-s_len])))  # 右移图像
                ls_image = np.append(ls_image,
                                     (np.append(row[s_len:], row[:s_len])))
            self.results['random_shift'].append(ls_image)
            self.results['random_shift'].append(rs_image)

#
# if __name__ == '__main__':
#     images, labels = train_images, train_labels
#
#     images = images[labels == 8][:25]
#     labels = labels[labels == 8][:25]
#
#     geo_augmenter = GeoAugmenter(images, labels)
#
#     geo_augmenter.random_rotate()
#     geo_augmenter.random_shift()
#
#     path = '../../Demo/MNIST'
#
#     visualize(images, os.path.join(path, 'origin_8.png'))
#
#     for f in geo_augmenter.results:
#         visualize(geo_augmenter.results[f], os.path.join(path, f + '.png'))
