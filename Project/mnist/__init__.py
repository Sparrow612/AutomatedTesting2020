import tensorflow.keras as keras
import os
from Project.mnist.strategies import *

mnist = keras.datasets.mnist
dataset = (x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Mnist dataset is ready!')


def top2Accu(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 2)  # k=2


def format_process(test, m_name):
    if 'dnn' in m_name:
        return test.reshape(-1, 784)
    else: return test.reshape((-1, 28, 28, 1))


root = '../../Models/MNIST'''
models = sorted(os.listdir(root))
idx = 1  # 选取几号模型

m = models[idx]
o = 'adam'
l = 'sparse_categorical_crossentropy'

model_path = os.path.join(root, m)
model = keras.models.load_model(model_path)
model.compile(optimizer=o, loss=l,
              metrics=[keras.metrics.sparse_categorical_accuracy, top2Accu,
                       keras.metrics.sparse_categorical_crossentropy])

# data = ImageDataset(train_data=dataset[0], test_data=dataset[1])
# data.preprocess(strategy[1], train=True)
# model.fit(x=data.train_images.reshape((-1, 28, 28, 1)), y=data.train_labels, epochs=10, verbose=1, shuffle=True,
#           batch_size=32, validation_split=0.1)
# model.save('vgg.hdf5')

score = model.evaluate(x=format_process(x_test, m), y=y_test, verbose=1)
print('model:', m)
print('optimizer:', o, '\n', 'loss:', l)
print('Origin')
print('Loss:', score[0])
print('Accuracy:', score[1])
