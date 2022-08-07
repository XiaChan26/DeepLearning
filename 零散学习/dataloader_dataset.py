"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

'''
    data.DataLoader(
    dataset,
    batch_size = 50,
    shuffle = False,
    sampler=None,
    batch_sampler = None,
    num_workers = 0,
    collate_fn = 
    pin_memory = False,
    drop_last = False,
    timeout = 0,
    worker_init_fn = None,
)

    torch.utils.data.DataLoader参数：
    (
    dataset (Dataset) – 待传入的数据集，也就是上面自己实现的myData。
    batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
    shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据，在每个epoch开始的时候，对数据进行重新排序。(默认: False).
    sampler (Sampler, optional) – 定义从数据集中提取样本的策略，即生成index的方式，可以顺序也可以乱序
    batch_sampler：类似于sampler，不过返回的是一个迷你批次的数据索引。
    num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
    collate_fn (callable, optional) –将一个batch的数据和标签进行合并操作。
    pin_memory (bool, optional) –设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
    timeout，是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0。
    )
'''

# 1、导入包的方式构建数据集
# dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# 2、自定义创建数据集

'''
    注释：任何自定义的小型数据集，都要继承torch.utils.data.Dataset，然后重写
    两个函数：__len__(self)和__getitem__(self, idx)
'''


class myDataset(Dataset):  # 需要导入包from torch.utils.data import Dataset
    def __init__(self):
        # 创建5*2的数据集
        self.data = torch.tensor([[1, 2], [3, 4], [4, 5], [2, 1], [3, 2]])
        self.label = torch.tensor([0, 1, 0, 1, 2])

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组方式返回

    # 获取数据集大小
    def __len__(self):
        return len(self.data)


data = myDataset()
print(f'data size is : {len(data)}')
print(data[1])  # 获取索引为1的data和label

# =========================torch.utils.data.DataLoader=======================================
'''torch.utils.data.Dataset通过__getitem__获取单个数据，如果希望获取批量数据、shuffle或者其它的一些操作，那么就要由torch.utils.data.DataLoader来实现了'''
# from torch.utils.data import DataLoader
data = myDataset()

my_loder = DataLoader(data, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
for step, train_data in enumerate(my_loder):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    Data, Label = train_data
    print("step:", step)
    print("data:", Data)
    print("label:", Label)

# ================================将数据加载到GPU===============================================
'''
    在Dataset和DataLoader的地方都可以实现把数据放入GPU，下面分别进行介绍。
'''
# 1、Dataset阶段把数据放入GPU
'''
    如果在此阶段把数据放入GPU，则此阶段必须把num_workers设置为0，要不然会报错。此阶段的操作需要在__getitem__中实现，实现过程大致如下。
'''


def __getitem__(self, index):
    data = torch.Tensor(self.Data[index])
    label = torch.IntTensor(self.Label[index])
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()
    return data, label


# 2、DataLoader阶段把数据放入GPU
'''
    这种实现方式就没有特别需要注意的地方，直接把tensor放入GPU即可，所以推荐使用这种实现方式，如下所示。
'''
data = myDataset()

my_loader = DataLoader(data, 2, shuffle=False, num_workers=0, drop_last=True)
for step, train_data in enumerate(my_loader):
    Data, Label = train_data
    # 把数据放在GPU中
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()
    print("step:", step)
    print("data:", Data)
    print("Label:", Label)
