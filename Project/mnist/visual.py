from Project.mnist import *
import tensorflow as tf
from matplotlib import pyplot as plt
from Project.mnist.strategies import *

db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db.shuffle(10000)
data = db.batch(9)
# 获得一个batch的数据
x_target, y_target = data.as_numpy_iterator().next()
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.title(y_target[i])
    plt.axis('off')
    plt.imshow(x_target[i], cmap=plt.get_cmap('gray'))
    plt.savefig('origin.png')

it = data.as_numpy_iterator()
for index, s in enumerate(strategy):
    x_target, y_target = it.next()
    for idx, img in enumerate(x_target):
        x_target[idx] = s(img)

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.title(y_target[i])
        plt.axis('off')
        plt.imshow(x_target[i], cmap=plt.get_cmap('gray'))
        plt.savefig('strategy{}'.format(index) + '.png')
