import torch

'''
    x.unsqueeze(dim=a)
    用途：进行维度扩充，在指定位置加上维数为1的维度
    参数设置：如果设置dim = a，就是在维度为a的位置进行扩充
'''
x = torch.tensor([1, 2, 3, 4])
print('x', x)
x1 = x.unsqueeze(dim=0)
print('x1', x1)
x2 = x.unsqueeze(dim=1)
print('x2', x2)

y = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
print('y', y)
y1 = y.unsqueeze(dim=0)
print('y1', y1)
y2 = y.unsqueeze(dim=1)
print('y2', y2)

'''
    x.squeeze(dim)
    用途：进行维度压缩，去掉tensor中维数为1的维度
    参数设置：如果设置dim=a，就是去掉指定维度中维数为1的
'''
z = torch.tensor([[[1], [2]], [[3], [4]]])
print('z:', z)
z1 = z.squeeze()
print('z1:', z1)
z2 = z.squeeze(2)
print('z2:', z2)
