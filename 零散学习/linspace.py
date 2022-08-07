import torch

# linspace(start, end, steps)参数设置
# start 开始值
# end 结束值
# steps 分割的点数（就是被分成多少份，15就是代表从1到10，被分成15份）
x = torch.linspace(1, 10, 15)
print(x)