import torch
from torchvision import transforms
a = torch.arange(1, 17)  # a's shape is (16,)

x = a.view(4, 4)  # output below
print(x)
x1 = a.view(-1, 4)
print(x1)
y = a.view(2, 2, 4)  # output below
print(y)