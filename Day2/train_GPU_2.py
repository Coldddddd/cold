import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 设置训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10(
    root="../dataset_chen", 
    train=True, 
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root="../dataset_chen", 
    train=False, 
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 打印数据集大小
print(f"训练数据集大小: {len(train_dataset)}")
print(f"测试数据集大小: {len(test_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义模型结构
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

# 实例化模型并转移到设备
model = CustomModel().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练设置
num_epochs = 10
train_steps = 0
test_steps = 0

# TensorBoard记录
writer = SummaryWriter("../logs_train")

# 记录训练开始时间
start_time = time.time()

for epoch in range(num_epochs):
    print(f"----- 第{epoch + 1}轮训练开始 -----")
    
    # 训练阶段
    model.train()
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps += 1
        if train_steps % 200 == 0:
            print(f"步骤 {train_steps}, 训练损失: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), train_steps)

    # 训练周期结束
    end_time = time.time()
    print(f"这一轮训练的时间: {end_time - start_time:.2f}秒")

    # 测试阶段
    model.eval()
    total_test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            
            # 计算正确预测的数量
            correct_predictions += (outputs.argmax(1) == labels).sum().item()

    # 输出测试结果
    accuracy = correct_predictions / len(test_dataset)
    print(f"测试损失: {total_test_loss:.4f}, 测试准确率: {accuracy * 100:.2f}%")
    
    # 记录测试结果
    writer.add_scalar("test_loss", total_test_loss, test_steps)
    writer.add_scalar("test_accuracy", accuracy, test_steps)
    test_steps += 1

    # 保存模型
    torch.save(model.state_dict(), f"model_save/custom_model_epoch_{epoch + 1}.pth")
    print("模型已保存")

# 关闭TensorBoard
writer.close()
