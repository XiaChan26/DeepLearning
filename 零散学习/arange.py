import torch
# torch.arange() 返回发小的一个张量
# 用法torch.arange(start=0, end, step=1, out=None)
'''
start：点集的起始值。默认值：0。
end：点集的最终值
step：每对相邻点之间的间隙。默认值：1。
'''
x = torch.arange(16).view((1, 1, 4, 4))

print(x)
y = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(y)
z = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(z)
