import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

cifar = keras.datasets.cifar100
dataset = (x_train, y_train), (x_test, y_test) = cifar.load_data()

print('Cifar dataset is ready!')


def top2Accu(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 2)  # k=2


root = '../../Models/CIFAR100'

models = sorted(os.listdir(root))
idx = 3  # 选取几号模型
m = models[idx]
o = 'adam'
l = 'sparse_categorical_crossentropy'

model_path = os.path.join(root, m)
# model_path = 'ResNet.h5'
model = keras.models.load_model(model_path)
# model.summary()
model.compile(optimizer=o, loss=l,
              metrics=[keras.metrics.sparse_categorical_accuracy, top2Accu,
                       keras.metrics.sparse_categorical_crossentropy])

# image_train = np.load('train_images.npy')
# label_train = np.load('train_labels.npy')
# model.fit(x=image_train/255., y=label_train, batch_size=100, epochs=50, shuffle=True, verbose=1,
#           callbacks=[EarlyStopping(monitor='loss', patience=10), TensorBoard(log_dir='./logs')])
# model.save('ResNet.h5')
score = model.evaluate(x=x_test / 255., y=y_test, verbose=1)
print('model:', m)
print('optimizer:', o, '\n', 'loss:', l)
print('Origin')
print('Loss:', score[0])
print('Accuracy:', score[1])
