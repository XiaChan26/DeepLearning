import torch

# 设置随机种子
# 保证每次运行的随机数是一致的

torch.manual_seed(0)
print(torch.randn(3))
print(torch.rand(3))

# 输出，再次运行时，输出的结果依旧如此
# tensor([0.4963, 0.7682, 0.0885])

# tensor([0.1320, 0.3074, 0.6341])
