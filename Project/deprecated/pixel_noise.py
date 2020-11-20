import random
import os
from Project.mnist import *
from collections import defaultdict
import numpy as np

img_shape = (28, 28)


def random_noise(img, n_rate=0.5, n_l=-64, n_r=64):
    """
    施加微小的随机扰动
    """
    img = img.flatten()
    res = img.copy().astype(dtype=np.int16)
    for idx in range(len(res)):
        rdn = random.random()
        if rdn > 1 - n_rate:
            n = random.randint(n_l, n_r)
            res[idx] += n
    res = np.clip(res, 0, 255).astype(dtype=np.uint8)
    return res.reshape(img_shape)


def change_background(img, new_bg=128):
    """
    改变一下图片的背景
    """
    img = img.flatten()
    res = np.zeros(img.shape)
    for idx in range(len(res)):
        res[idx] = max(img[idx], new_bg)
    res = np.clip(res, 0, 255)
    return res.reshape(img_shape)


def bright_adjust(img, k_down=0.75, k_up=1.25, b_down=64, b_up=64):
    """
    图像亮度、对比度调整
    :param img: 源图像
    :param k_down:对比度系数下限
    :param k_up:对比度系数上限
    :param b_down:亮度增值上限
    :param b_up:亮度增值下限
    :return:调整后的图像
    """


def salt_and_pepper(img, prob=0.01):
    """
    椒盐噪声
    :param img: 源图像
    :param prob: 噪声率，白黑均分
    :return: 加噪后的图像
    """
    img = img.flatten()
    output = np.zeros(img.shape, np.float)
    threshold = 1 - prob
    for i in range(img.shape[0]):
        rdn = random.random()  # 随机生成0-1之间的数字
        if rdn < prob:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
            output[i] = 0
        elif rdn > threshold:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
            output[i] = 255
        else:
            output[i] = img[i]  # 其他情况像素点不变
    return output.reshape(img_shape)


def gasuss_noise(img, mean=128, var=0.001):
    """
        添加高斯噪声
        img: 原始图像
        mean : 均值
        var : 方差越大，噪声越大
    """
    noise = np.random.normal(mean, var ** 0.5, img.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = img + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    out = np.clip(out, 0, 255)  # clip函数将元素的大小限制在了0和1之间了，小于的用low_clip代替，大于1的用1代替
    return out


funcs = [bright_adjust, gasuss_noise, salt_and_pepper, random_noise, change_background]


def augment(imgs, ls):
    """
    批量数据增强，调用上面的所有方法，将数据扩充为原来的5倍
    """
    a_images = defaultdict(list)
    a_labels = defaultdict(list)
    for idx in range(len(imgs)):
        for func in funcs:
            a_images[str(func)].append(
                func(imgs[idx].copy())
            )  # 传递数组的拷贝，这样做更加安全
            a_labels[str(func)].append(ls[idx])
    return a_images, a_labels


if __name__ == '__main__':
    out_path = '../../Demo/MNIST'

    images, labels = train_images, train_labels
    images = images[labels == 7][:25]

    visualize(images, os.path.join(out_path, 'origin_7.png'))

    imgs, labels = augment(images, labels)
    for f in imgs:
        fname = f.split()[1]
        visualize(imgs[f], os.path.join(out_path, fname + '.png'))
