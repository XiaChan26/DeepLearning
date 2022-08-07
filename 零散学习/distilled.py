# 在MNIST数据集上，从头训练学生网络，知识蒸馏训练学生网络，比较性能
# 导入工具包
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# 设置随机种子，便于复现，每次的
torch.manual_seed(0)
# 检测GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 使用cuDNN加速卷积运算，放在使用GPU的地方，英伟达专门为
torch.backends.cudnn.benchmark = True
# 载入数据集
train_dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# 载入测试集
test_dataset = torchvision.datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# 生成dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# ==============================================教师模型=====================================================
class TeacherModel(nn.Module):          # 定义网络！！！！
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        # 会用到的层，先定义好，方便forward函数使用！
        self.relu = nn.ReLU()
        # 三层Linear
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):           # 这才是模型层！！！！！
        # 这个是使用上一个在__init__中定义好的，因为在上一层
        # 类似self.fc1，是针对于整个TeacherModel类，可以利用self拿过来用
        # 这个部分就是模型层！！！
        x = x.view(-1, 784)
        x = self.fc1(x)
        # 加了dropout
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        # 加了dropout
        x = self.dropout(x)
        x = self.relu(x)

        # 第三层
        x = self.fc3(x)

        return x


# 实例化
model = TeacherModel()
# 加载到GPU中运行（载入运行设备）
model = model.to(device)
# summary(model)输出各层的参数状况！！！！
summary(model)
# 交叉熵分类损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器，学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoch = 6
# =========================训练过程和评估过程！！！！！========================================
for epoch in range(epoch):
    # 启动batch normalization和drop out
    model.train()
    # model.train()的作用：启用batch normalization和drop out，这里是启用drop out。
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # 前向预测
        preds = model(data)
        loss = criterion(preds, targets)    # criterion = nn.CrossEntropyLoss()
        # 反向传播，优化权重
        optimizer.zero_grad()       # 把梯度置零，也就是把loss关于weight的导数变成0
        loss.backward()             # 反向传播求梯度
        optimizer.step()            # 更新所有参数

    # 测试集上评估模型性能！！================================================
    model.eval()        # 作用：沿用batch normalization的值，并不使用drop out。
    num_correct = 0
    num_samples = 0

    with torch.no_grad():       # 只想看一下训练结果，而不想通过验证集来更新网络，可以使用with torch.no_gard():
        for x, y in test_loader:    # 将验证机加载到GPU上
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    model.train()
    print('T_Epoch:{}\t Accuracy:{:.4f}'.format(epoch + 1, acc))

teacher_model = model


# ======================================学生模型=================================================
class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        # 20个神经元
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.dropout(x)
        x = self.relu(x)

        # 第三层
        x = self.fc3(x)

        return x


# 从头训练一下学生模型
model = StudentModel()
model = model.to(device)

# 交叉熵分类损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器，学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoch = 3
for epoch in range(epoch):
    model.train()
    # 训练集上训练模型权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # 前向预测
        preds = model(data)
        loss = criterion(preds, targets)
        # 反向传播，优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试集上评估模型性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    model.train()
    print('S_Epoch:{}\t Accuracy:{:.4f}'.format(epoch + 1, acc))
student_model_scratch = model

# ===============================================用知识蒸馏训练学生模型================================================

# 准备训练好的教师模型
teacher_model.eval()

# 准备新的学生模型
model = StudentModel()
model = model.to(device)
model.train()

# 蒸馏温度
temp = 6

# hard_loss,CrossEntropyLoss普通的分类交叉熵损失函数
hard_loss = nn.CrossEntropyLoss()
# hard_loss权重
alpha = 0.3

# soft_loss，KLDivLoss   KL散度，也差不多是个交叉熵损失函数
soft_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    # 训练集上训练模型权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # 教师模型预测
        with torch.no_grad():
            teacher_preds = teacher_model(data)

        # 学生模型预测
        student_preds = model(data)
        # 计算hard_loss
        student_loss = hard_loss(student_preds, targets)

        # 计算蒸馏后的预测结果及soft_loss
        ditillation_loss = soft_loss(
            F.softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )

        # 将hard_loss和soft_loss加权求和!!!!!!!
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss

        # 反向传播，优化权重
        optimizer.zero_grad()           # 梯度初始化为零
        loss.backward()                 # 反向传播求梯度
        optimizer.step()                # 更新所有参数

    # 测试集上评估模型性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    model.train()
    print('D_Epoch:{}\t Accuracy:{:.4f}'.format(epoch + 1, acc))
