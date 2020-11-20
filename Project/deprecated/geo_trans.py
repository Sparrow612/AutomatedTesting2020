from Project.cifar import *
from PIL import Image
import random
import cv2


def random_crop(image, crop_shape, padding):
    """
    随机擦除，我自己的版本
    :param image: 源图像
    :param crop_shape: 切割图像尺寸，也就是最终输出的图像尺寸
    :param padding: 填充长度，int
    :return:
    """
    shape_aft_pad = np.shape(image)
    shape_aft_pad = (shape_aft_pad[0] + 2 * padding, shape_aft_pad[1] + 2 * padding)
    npad = ((padding, padding), (padding, padding))  # 上下左右各填充padding
    image_pad = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    nh = random.randint(0, shape_aft_pad[0] - crop_shape[0])
    nw = random.randint(0, shape_aft_pad[1] - crop_shape[1])
    image_crop = image_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return image_crop


def random_rotate_zoom(img, limit=45, z_rate=(0.7, 1.3)):
    """
    BorderMode参考：https://www.cnblogs.com/my-love-is-python/archive/2004/01/13/10390555.html
    :param z_rate:
    :param img:
    :param limit:
    :return:
    """
    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = random.uniform(-limit, limit)
    zoom = random.uniform(z_rate[0], z_rate[1])
    M = cv2.getRotationMatrix2D(center_coordinate, angle, zoom)
    rotate_img = cv2.warpAffine(img, M, (rows, cols), borderMode=cv2.BORDER_REFLECT)
    return rotate_img


def random_rotate_v2(img, limit=45):
    img = Image.fromarray(img)
    angle = random.uniform(-limit, limit)
    rotate_img = img.rotate(angle)
    return np.array(rotate_img)


# img = cv2.imread('0.png')
# img = random_rotate_zoom(img)
# cv2.imwrite('2.png', img)
