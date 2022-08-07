# 池化层的是实现--------降维！

import torch
from torch import nn


# 这里的mode参数可以填max 也可以天avg，二维：pool2d


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    # 初始化一个池化过后的空tensor,赋值为0，获取X的行数和列数，计算出池化之后的矩阵行和列
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()  # 最大池化，max() 返回给定参数的最大值
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()  # 平均池化，mean() 求平均值
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
q = pool2d(X, (2, 2))
# 被池化的原始矩阵
print('x1', X)
# 池化之后的矩阵
print('q', q)

