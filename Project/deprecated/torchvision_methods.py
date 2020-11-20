import torchvision.transforms as tf
from PIL import Image
import numpy as np
import cv2

"""
RGB格式
white: (255, 255, 255)
black: (0, 0, 0)
grey: (125, 125, 125)

本文件中的方法主要为调用API实现，包含一些方法的组合使用
"""

# tf.RandomCrop(32, padding=4, pad_if_needed=True),  # 随机裁减
# tf.RandomVerticalFlip(),  # 随机垂直翻转
# tf.RandomHorizontalFlip(),  # 随机水平翻转
# tf.Normalize((0.4914, 0.4822, 0.5165), (0.2023, 0.1994, 0.2010)),  # 正则化

transformer_erase_black = tf.Compose([
    tf.ToTensor(),
    tf.RandomErasing(),  # 随机擦除
    tf.ToPILImage(),
])

transformer_erase_grey = tf.Compose([
    tf.ToTensor(),
    tf.RandomErasing(value=0.5),  # 随机擦除
    tf.ToPILImage(),
])

transformer_erase_white = tf.Compose([
    tf.ToTensor(),
    tf.RandomErasing(value=1),  # 随机擦除
    tf.ToPILImage(),
])

transformer_erase_random = tf.Compose([
    tf.ToTensor(),
    tf.RandomErasing(value='random'),  # 随机擦除
    tf.ToPILImage(),
])

transformer_VFlip = tf.RandomVerticalFlip()
transformer_HFlip = tf.RandomHorizontalFlip()
transformer_Crop = tf.RandomCrop(32, padding=4, pad_if_needed=True)
transformer_toGrey = tf.Grayscale()

transformer_norm = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.4914, 0.4822, 0.5165), (0.2023, 0.1994, 0.2010)),
    tf.ToPILImage(),
])

transformer_NormAndErase = tf.Compose([
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    tf.RandomErasing(),
    tf.ToPILImage(),
])  # 源代码中的使用案例

funcs = [transformer_erase_black, transformer_erase_grey, transformer_erase_white, transformer_erase_random,
         transformer_toGrey, transformer_VFlip, transformer_HFlip, transformer_Crop, transformer_norm,
         transformer_NormAndErase]

img = Image.open('../mnist/0.png').convert('RGB')
t = tf.GaussianBlur((3, 3))
out = t(img)
out = np.array(out)
cv2.imwrite('1.png', out)
# 下面部分为效果示例代码
# img = Image.open('/Users/chengrongxin/PycharmProjects/AutomatedTesting2020/Demo/MNIST/change_background.png').convert(
#     'RGB')
# res = np.array(transformer_Crop(img))
# cv2.imwrite('0.png', res)
