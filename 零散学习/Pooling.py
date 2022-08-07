'''
实际图像里，我们所感兴趣的物体不会总是出现在同一个地方，这些物品
一定会出现在不同的像素位置，因此导致同一个边缘的输出会在总的卷积输出Y
的不同同位置，造成模式识别的不便
而池化层的提出是为了缓解卷积层对位置的过度敏感性
'''

''' 二维最大池化函数和平均池化层 '''

''' 池化层的是实现 '''

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

# X是被池化的矩阵


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
q = pool2d(X, (2, 2))
# 被卷积的原始矩阵
print('x1', X)
# 打印pool2d(X, (2, 2))最大池化之后的结果
print('最大池化之后的', q)
# %%

'''填充和步幅'''
'''
和卷积层一样，池化层也可以用填充和步幅来改变输出形状
这里使用nn模块中的MaxPool2d池化层
'''
# 首先构造一个形状维（1，1，4，4）的输入数据，前两个维度维批量和通道
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)

# 默认情况下MaxPool2d的步幅和池化层窗口形状相同，即3*3的输入，池化层的步幅也是3*3
pool2d = nn.MaxPool2d(3)
w = pool2d(X)
print('w', w)
# 手动指定非正方形的池化窗口，并指定高和宽上的填充和步幅
# (2, 4)池化核为2*4；
# padding=(1, 2)上下各一行0，左右各两列0；
# stride=(2, 3)步长，行内移动3，列移动2
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
r = pool2d(X)
print('r', r)
# %%

'''多通道'''
'''
池化层对于多通道的处理方式和卷积层不同，卷积层是将输入按权计算再按通道相加
而池化层没有后一步，也就是说池化层的输入通道和输出通道是一样的
'''
# 构造通道为2的输入
X = torch.cat((X, X + 1), dim=1)
print('t', X.shape)
# 池化后通道数还是2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
