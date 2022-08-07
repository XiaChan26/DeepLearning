import torch
import torch.nn as nn

'''二维卷积层'''

'''一、交叉相关运算(卷积运算)'''


def corr2d(X, K):
    h, w = K.shape  # .shape 是返回目标K的长和宽，此时h=2,w=2，卷积核
    # print('h,w：', h, w)
    '''
        X.shape[0]输出的是矩阵Y的行数，X.shape[1]输出的是矩阵Y的列数，此时分别为3,3
        得到此时卷积出来的矩阵为3-2+1=2，3-2+1=2。为2*2的矩阵
        这里根据跟指定的X，Y计算输出结果的形状，并初始化该形状的矩阵元素全为0
    '''
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    for i in range(Y.shape[0]):  # 遍历Y的长度，将卷积出来的数放入对应的位置
        for j in range(Y.shape[1]):  # 遍历Y的宽度，将卷积出来的数放入对应的位置
            # 矩阵相乘在相加
            # [i: i + h, j: j + w]数组切片，第一轮切片为为2*2的一个大小
            # 二维数组切片
            # [i: i + h, j: j + w]表示[0:2，0:2]，第一维取下标2之前的，即（0，1两行）。列取（0，1两列）
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()  # Y[i, j]指此时对应的位置，后面是对应的数字
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(X)
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))
# 输出
# tensor([[19., 25.],
#         [37., 43.]])


'''图像中物体的边缘检测'''

# 构造一个简单的图像，中间为黑，两边为白
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)


# 构造一个简单的1*2的卷积核
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)

