import pickle

from Project.cifar import *
import tensorflow as tf
from matplotlib import pyplot as plt
from Project.cifar.strategies import *


def load_labels_name(filename):
    """使用pickle反序列化labels文件，得到存储内容
        cifar10的label文件为“batches.meta”，cifar100则为“meta”
        反序列化之后得到字典对象，可根据key取出相应内容
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


label_dict = load_labels_name('../../Data/CIFAR-100/cifar-100-python/meta')['fine_label_names']

db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db.shuffle(10000)
data = db.batch(9)
# 获得一个batch的数据
it = data.as_numpy_iterator()
x_target, y_target = it.next()
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.title(label_dict[y_target[i][0]])
    plt.axis('off')
    plt.imshow(x_target[i])
    plt.savefig('./imgs/origin.png')

for index, s in enumerate(strategy):
    x_target, y_target = it.next()
    for idx, img in enumerate(x_target):
        r = s(img)
        if isinstance(r, dict):
            x_target[idx] = r['image']
        else:
            x_target[idx] = np.array(r)

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.title(label_dict[y_target[i][0]])
        plt.axis('off')
        try:
            plt.imshow(x_target[i])
        except TypeError:
            print(index)
        plt.savefig('./imgs/strategy{}'.format(index) + '.png')
