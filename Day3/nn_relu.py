import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

# 加载CIFAR10数据集
test_set = torchvision.datasets.CIFAR10(root='dataset_chen', train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(test_set, batch_size=64)

# 定义输入张量
input_tensor = torch.tensor([[1, -0.5], [-1, 3]])
input_tensor = input_tensor.view(-1, 1, 2, 2)
print(input_tensor.shape)

# 定义自定义模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(x)

# 初始化模型
model = SimpleModel()

# 设置TensorBoard记录器
tb_writer = SummaryWriter('logs_react')
step = 0
for batch in data_loader:
    images, labels = batch
    tb_writer.add_images('input_images', images, global_step=step)
    predictions = model(images)
    tb_writer.add_images('output_images', predictions, global_step=step)
    step += 1

tb_writer.close()

# 测试单一输入
output_tensor = model(input_tensor)
print(output_tensor)
