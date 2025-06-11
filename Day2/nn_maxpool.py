import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR-10数据集
test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_chen", train=False,
    transform=torchvision.transforms.ToTensor(), download=True
)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义模型
class PoolingModel(nn.Module):
    def __init__(self):
        super(PoolingModel, self).__init__()
        self.pooling_layer = MaxPool2d(kernel_size=3)

    def forward(self, x):
        return self.pooling_layer(x)

# 初始化模型
model = PoolingModel()

# 设置TensorBoard
writer = SummaryWriter("maxpool_logs")
step = 0

# 训练过程
for images, labels in test_loader:
    # 记录输入图像
    writer.add_images("Input Images", images, step)
    
    # 前向传播
    pooled_output = model(images)
    
    # 记录池化后的输出
    writer.add_images("Pooled Output", pooled_output, step)
    
    step += 1

# 关闭TensorBoard
writer.close()
