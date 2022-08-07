# 线性回归模型

import torch
import matplotlib
import numpy as np
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
import random

# 查看版本
# print(torch.__version__)

# 使用线性模型生成一个数据集，生成一个1000个样本的数据集

# 设置输入特征数量，两个特征
num_inputs = 2
# 设置样本数量
num_examples = 1000
# 设置真实的权重和偏差以生成相应的标签
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 1000*2的矢量
labels = true_w[0] * features[:, 0] + true_w[0] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# 使用图像来展示生成的数据
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# 读取数据集


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机读取10个样本
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足以整个批次
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)  # 梯度的附加操作
b.requires_grad_(requires_grad=True)


def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数（均方误差损失函数）
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化函数（小批量随机梯度下降）
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 使用.data可以在没有渐变轨迹的情况下操作参数


# 训练
# 超参数初始化，超参数是需要人为设置的参数
lr = 0.03  # 学习率
num_epochs = 5  # 训练周期

net = linreg
loss = squared_loss

# 训练
for epoch in range(num_epochs):  # 训练重复num_epochs次
    # 每个epoch中，所有在数据集中的样本都被使用一次

    # X是特征，y是批量样本的标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        # 计算批量样本损失的梯度
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w, b], lr, batch_size)
        # 重置参数梯度
        w.grad.data.zero_()  # 参数梯度清零，防止参数累加
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d,loss %f' % (epoch + 1, train_l.mean().item()))
