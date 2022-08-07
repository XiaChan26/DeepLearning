import torch
import torch.nn as nn

# 当我们指定了设备之后，就需要将模型加载到相应的设备中，此时就需要使用
# device = model.to(device)将模型加载入设备中
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model部分
class TeacherModel(nn.Model):
    #     ---该模型并不完整，仅作为尝试使用加载到运行设备写法----
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()

    def forword(self, x):
        return x


model = TeacherModel
device = model.to(device)  # 此处的model是在model部分定义的

'''
device1 = torch.device('cpu')
device2 = torch.device('cuda')  # 此处是cuda，而不是gpu
print(device)
print(device1)
print(device2)
'''
