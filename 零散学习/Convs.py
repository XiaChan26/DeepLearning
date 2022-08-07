# 卷积层实现
import torch
import torch.nn as nn

'''二维卷积层'''
'''该函数计算二维互相关运算'''


def corr2d(X, K):
    h, w = K.shape
    # 首先构造出运算过后的矩阵形状并用0填充
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

# %%

'''图像中物体的边缘检测'''
# 构造一个简单的图像，中间为黑，两边为白
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

# 构造一个简单的1*2的卷积核
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)

# %%

'''卷积层的构造'''
'''
二维卷积层将输入和卷积和做互相关运算，并加上一个标量偏差来得到输出。
卷积层模型参数包括卷积核和标量偏差
最后得出的结果和上面的[1,-1]的卷积类似
'''


class Conv2D(nn.Module):
    # 在构造函数中声明权重和偏差
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        # 随机初始化参数
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


conv2d = Conv2D(kernel_size=(1, 2))
step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.zero_()
    # if (i + 1) % 5 == 0:
    print('Step %d, loss %.3f' % (i + 1, l.item()))
print(conv2d.weight.data, conv2d.bias.data)

# %%

'''这里展示了填充'''
'''填充指的是再高和宽两侧填充元素（通常填充零元）'''


# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1,1)代表批量大小和通道数，均为1,这里的view函数相当于给他们多增加俩维度
    # print((1,1)+X.shape)
    X = X.view((1, 1) + X.shape)
    # print(X)
    Y = conv2d(X)
    # print(Y)
    # 排除不关心的前两维：批量和通道，这里的view函数相当于只取i最后俩维度
    return Y.view(Y.shape[2:])


# 创建一个高和宽都为3的卷积层，在高和宽两侧的填充数分别为1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X = torch.randn(4, 4)
Y = comp_conv2d(conv2d, X)
# print(Y)

# 创建一个高和宽都为5和3的卷积层，在高和宽两侧的填充数分别为2，1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
X = torch.randn(4, 4)
Y = comp_conv2d(conv2d, X)

# %%

'''这里展示步幅'''
# 在创建卷积层的时候添加stride参数就是步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape

# %%

'''
多输入通道，也就是上面的通道数那个维度数大于一的时候
这时候就需要构造一个和通道数相等的卷积层然后用不同的通道对应不同的
卷积运算，最后再相加
'''


def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


# 验证一下，两个通道，两个卷积filter
X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
corr2d_multi_in(X, K)

# %%

'''
多输出通道，
'''


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关运算
    # stack函数是用来连接tensor的
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# 这里需要构造一个输出通道*输入通道*高*宽的卷积层
K = torch.stack([K, K + 1, K + 2])
corr2d_multi_in_out(X, K)

# %%
'''
1*1卷积层
调整网络层之间的通道数来控制模型复杂度
'''
'''
可以想一想下，1*1卷积层和其他卷积比起来缺失了可以识别高和宽度维度上的相邻
元素构成的模式的功能。
所以它的运算主要发生在通道维度上，它可以起到增大或者减小通道维的值。
比如：输入是3*3*3的图像，对它进行两个1*1的卷积运算得到的输出是3*3*2的图像矩阵。
输出中的每个元素来自于输入中的高和宽上相同位置的元素再不同通道上按1*1给的卷积进行按权累加
'''


def corr2d_multi_in_out_1x1(X, K):
    # 通道数，高，宽
    c_i, h, w = X.shape
    # 1*1卷积通道数
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


# torch.randn和torch.rand一个是标准正态分布，一个是均匀分布
X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1, Y2)
