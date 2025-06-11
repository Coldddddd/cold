import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 数据集准备
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

# 输出数据集大小
train_size = len(train_dataset)
test_size = len(test_dataset)
print(f"训练集大小: {train_size}")
print(f"测试集大小: {test_size}")

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
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
        return self.conv_block(x)

# 实例化模型并迁移至GPU
model = SimpleCNN()
if torch.cuda.is_available():
    model = model.cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练参数
epochs = 10
train_steps = 0
test_steps = 0

# TensorBoard日志记录
log_writer = SummaryWriter("../logs_train")

# 训练开始时间
start_time = time.time()

for epoch in range(epochs):
    print(f"--- 第{epoch+1}轮训练开始 ---")
    
    # 训练阶段
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_steps += 1
        if train_steps % 500 == 0:
            print(f"训练步骤 {train_steps}，损失: {loss.item()}")
            log_writer.add_scalar("Train_Loss", loss.item(), train_steps)

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / test_size
    print(f"测试集损失: {avg_test_loss}, 测试集准确率: {accuracy * 100:.2f}%")

    log_writer.add_scalar("Test_Loss", avg_test_loss, test_steps)
    log_writer.add_scalar("Test_Accuracy", accuracy, test_steps)
    test_steps += 1

    # 保存每轮模型
    torch.save(model.state_dict(), f"saved_models/epoch_{epoch+1}_model.pth")
    print("模型已保存")

# 关闭TensorBoard日志
log_writer.close()

# 输出总训练时间
end_time = time.time()
print(f"训练完成，总时间: {end_time - start_time:.2f}秒")
