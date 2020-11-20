import os
import torch.utils.data as dt
import numpy as np


class ImageDataset(dt.Dataset):
    def __init__(self, train_data, test_data):
        super(ImageDataset, self).__init__()
        # 提取训练集和测试集
        self.train_images, self.train_labels = train_data
        self.test_images, self.test_labels = test_data

    def preprocess(self, augment, train=False):
        """
        数据预处理，即数据增强，在这里的策略是
        :return: 生成新的测试数据
        """
        n = len(self.test_labels)
        tar = self.train_images if train else self.test_images
        for idx in range(n):
            r = augment(tar[idx])
            if isinstance(r, dict):
                r = r['image']
            tar[idx] = np.array(r)
            print('图像{}扩增完成'.format(idx))

    def __getitem__(self, index: int, train=True):
        if train:
            img = self.train_images[index]
        else:
            img = self.test_images[index]
        return img

    def __len__(self, train=True) -> int:
        return len(self.train_labels) if train else len(self.test_labels)

    def extract_train_data(self):
        return self.train_images, self.train_labels

    def extract_test_data(self):
        return self.test_images, self.test_labels

    def save(self, path, train=True):
        if train:
            np.save(os.path.join(path, 'mnist_train_images.npy'), self.train_images)
            np.save(os.path.join(path, 'mnist_train_labels.npy'), self.train_labels)
        else:
            np.save(os.path.join(path, 'mnist_test_images.npy'), self.test_images)
            np.save(os.path.join(path, 'mnist_test_labels.npy'), self.test_labels)
        print('数据保存成功！开始评估')
