import random
from Project.cifar import *
import cv2

"""
改变图像的一些基本属性：
亮度，对比度，饱和度，灰度
"""


# 随机亮度、随机对比度
def bright_adjust(img, k_down=0.75, k_up=1.25, b_down=-64, b_up=64):
    """
    :param img:
    :param k_down:
    :param k_up:
    :param b_down:
    :param b_up:
    :return:
    """
    # 对比度调整系数
    slope = random.uniform(k_down, k_up)
    # 亮度调整系数
    bias = random.uniform(b_down, b_up)
    img_copy = img.copy()
    avg = np.mean(img_copy.reshape(-1, 3), axis=0)
    # 图像亮度和对比度调整
    res = (img - avg) * slope + avg + bias
    res = np.clip(res, 0, 255)
    return res.astype(dtype=np.uint8)


def satur_adjust(img, satur_bound=(0, 255)):
    """
    饱和度
    :param img:
    :param satur_bound:
    :return:
    """
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    saturation = random.randint(satur_bound[0], satur_bound[1])
    hlsImg[:, :, 2] += saturation
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 255] = 255
    res = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return res


def toGrey(img):
    """
    算法参考链接：https://blog.csdn.net/giantpoplar/article/details/47295979
    :param img: RGB图像
    :return: 灰度图像
    """
    shape = img.shape
    constant = np.array([38 / 128, 75 / 128, 15 / 128]).T
    res = np.zeros(shape[:2])
    for row in range(shape[0]):
        for col in range(shape[1]):
            gray = np.dot(img[row][col], constant)
            res[row][col] = gray
    return res


class Compose:
    """
    组合使用上面的方法，模仿tf.Compose()
    灰度方法不能和saturation_adjust同时出现
    """

    def __init__(self, funcs):
        self.functions = funcs

    def __call__(self, img):
        for f in self.functions:
            img = f(img)
        return img


composer = Compose([
    bright_adjust, satur_adjust
])

# 测试用代码
img = cv2.imread('/Demo/MNIST/gasuss_noise.png')
res = toGrey(img)
cv2.imwrite('../mnist/0.png', res)

