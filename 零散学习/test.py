import torch


def corr2d(X, K):
    h, w = K.shape
    # 首先构造出运算过后的矩阵形状并用0填充
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


'''图像中物体的边缘检测'''
# 构造一个简单的图像，中间为黑，两边为白
# 构造一个6*8并且填充为1的矩阵
X = torch.ones(6, 8)
# 打印 X（内部数字为1）矩阵
print('X', X)
# 选取特定维度，令其值为0
# 作用于所有行，列从下标范围是2-6（不包含下标为6的列），并设置为0
X[:, 2:6] = 0
# 打印构建好的矩阵
print(X)

# 构造一个简单的1*2的卷积核
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)
