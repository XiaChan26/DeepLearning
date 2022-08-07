import torch

# mul() 对两个张量进行逐元素乘法
a = torch.rand(4, 1)
print(a)
b = torch.rand(1, 4)
print(b)
c = torch.mul(a, b)
print(c)


d = torch.tensor([[1], [3], [2], [1]])
print(d)
f = torch.tensor([2, 3, 1, 3])
print(f)
e = torch.mul(f, d)
print(e)

g = torch.tensor([[1, 2], [3, 4]])
print(g)
h = torch.tensor([[2, 1], [3, 4]])
print(h)
i = torch.mul(g, h)
print(i)