from torchvision import transforms
from Project.cifar.noise import *
import albumentations as am
from Project.cifar.random_erase import RandomErasing
from Project.cifar import *

# 下面的神秘代码来自ImageNet数据集算出的图像数据RGB均值和标准差，可以认为等同于自然界照片的均值、标准差
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])
# cifar数据集的形状
shape = (32, 32)

eraser = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_mean, std=rgb_std),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(shape, pad_if_needed=True, padding_mode='reflect'),
    RandomErasing(),  # self-made version
    transforms.Normalize(mean=-rgb_mean / rgb_std, std=1 / rgb_std),
    transforms.ToPILImage()
])  # 策略1，随机擦除法，参考论文'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896


class OneOf:
    """
    随机选择策略列表中的一种策略增强图像
    建议把所有加噪声的方法加入此类
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        n = len(self.transforms)
        idx = np.random.randint(0, n - 1)
        t = self.transforms[idx]
        return t(image=img)


rand_weather = OneOf([
    am.RandomRain(),
    am.RandomFog(),
    am.RandomSnow(),
])  # 策略2，随机天气，自然界可能随机发生一些天气现象，增强模型对天气现象的抗干扰能力，参考链接https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

sharpOrBlur = OneOf([
    am.GaussianBlur(),  # 高斯模糊在mnist中我已经提供了一个自己实现的版本，这里选择调库
    am.IAASharpen(),
])

noiser = OneOf([
    GaussNoise(),
    SaltAndPepper(),
])

strategy = [eraser, rand_weather, sharpOrBlur, noiser]

# res_images = np.zeros(shape=(100000, 32, 32, 3))
# res_labels = np.zeros(shape=(100000,))
#
# for epoch in range(2):
#     for idx, img in enumerate(x_train):
#         index = epoch * 50000 + idx
#         r = np.array(eraser(img))
#         res_images[index] = r
#         res_labels[index] = y_train[idx]
#         print('图像{}扩增完成'.format(idx))
# np.save('train_images.npy', res_images)
# np.save('train_labels.npy', res_labels)
