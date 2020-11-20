from Project.cifar.strategies import *
from Project.dataset import ImageDataset

for idx, s in enumerate(strategy):
    path = os.path.join('data', '{}/'.format(idx))
    files = sorted(os.listdir(path))
    if len(files):
        print('从已有数据集中抽取...')
        x_test = np.load(os.path.join(path, files[0]))
        # reshape(10000, -1)
        y_test = np.load(os.path.join(path, files[1]))
        print('读取成功！')
        score = model.evaluate(x=x_test/255., y=y_test, verbose=1)
    else:
        data = ImageDataset(train_data=dataset[0], test_data=dataset[1])
        print('开始新一轮扩增...')
        data.preprocess(s)
        data.save(path, train=False)  # 存下test数据集的扩增效果
        print('新的测试数据生成成功！使用策略:', idx)
        score = model.evaluate(x=data.test_images/255., y=data.test_labels, verbose=1)
    print('Strategy{}'.format(idx))
    print('Loss:', score[0])
    print('Accuracy:', score[1])
