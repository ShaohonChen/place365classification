import argparse

import paddle
import numpy as np
from paddle.vision import transforms

from datasets import Place365Dataset256


def get_arg():
    parser = argparse.ArgumentParser(description='train place365')
    parser.add_argument('--data_root', type=str, default='./place365/', help='datasets root')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    # parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    return parser.parse_args()


args = get_arg()
transfrom_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4, 4. / 3)),
    transforms.Normalize(mean=[127.5], std=[127.5]),
    transforms.ToTensor()

])

transfrom_val = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.Normalize(mean=[127.5], std=[127.5]),
    transforms.ToTensor()
])

# transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_dataset = Place365Dataset256(args.data_root, mode='train', transform=transfrom_train)
test_dataset = Place365Dataset256(args.data_root, mode='val', transform=transfrom_val)

# 模型组网并初始化网络
network = paddle.vision.models.resnet50(pretrained=True)
network.fc = paddle.nn.Linear(2048, 365)
model = paddle.Model(network)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 模型训练
model.fit(train_dataset, epochs=args.epoch, batch_size=args.batch, verbose=1)
# 模型评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

# 保存模型
model.save('./save/place365')
# 加载模型
model.load('save/place365')

# 从测试集中取出一张图片
img, label = test_dataset[0]
# 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)

# 执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
out = model.predict_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))
# 可视化图片
# from matplotlib import pyplot as plt
# plt.imshow(img[0])
