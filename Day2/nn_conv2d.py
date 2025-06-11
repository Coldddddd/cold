import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载数据集
cifar10_test = torchvision.datasets.CIFAR10(
    root="./dataset_chen", train=False, 
    transform=torchvision.transforms.ToTensor(), download=True
)
data_loader = DataLoader(cifar10_test, batch_size=64)

# 定义神经网络模型
class CHEN(nn.Module):
    def __init__(self):
        super(CHEN, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3)  # 卷积层，输入3通道，输出6通道，卷积核大小3

    def forward(self, x):
        return self.conv(x)

# 初始化模型
model = CHEN()
print(model)

# 初始化TensorBoard
writer = SummaryWriter("conv_logs")
step = 0

# 训练循环
for imgs, targets in data_loader:
    output = model(imgs)

    # 添加输入图像
    writer.add_images("Input Images", imgs, step)

    # 将输出调整为合适的形状并添加到TensorBoard
    reshaped_output = output.view(-1, 3, 30, 30)  # reshape层，调整为(-1, 3, 30, 30)
    writer.add_images("Conv Output", reshaped_output, step)
    
    step += 1
