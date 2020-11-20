from Project.dataset import ImageDataset
from Project.mnist import *
from Project.mnist.strategies import *

for idx, s in enumerate(strategy):
    path = os.path.join('data', '{}/'.format(idx))
    files = sorted(os.listdir(path))
    if len(files):
        print('从已有数据集中抽取...')
        x_test = np.load(os.path.join(path, files[0]))
        y_test = np.load(os.path.join(path, files[1]))
        print('读取成功！')
        score = model.evaluate(x=format_process(x_test, m), y=y_test, verbose=1)
    else:
        data = ImageDataset(train_data=dataset[0], test_data=dataset[1])
        print('开始新一轮扩增...')
        data.preprocess(s)
        data.save(path, train=False)  # 存下test数据集的扩增效果
        print('新的测试数据生成成功！使用策略:', idx)
        score = model.evaluate(x=format_process(data.test_images, m), y=data.test_labels, verbose=1)
    print('Strategy{}'.format(idx))
    print('Loss:', score[0])
    print('Accuracy:', score[1])
