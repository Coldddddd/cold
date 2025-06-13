import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

# 设备设置
device = torch.device("cpu")

# 数据预处理流水线
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CIFAR-10 数据集加载
train_set = torchvision.datasets.CIFAR10(root='dataset_chen', train=True, transform=data_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='dataset_chen', train=False, transform=data_transform, download=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 加载ResNet18并修改输出层
def create_model():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR-10的10分类
    return model.to(device)

# 定义训练函数
def run_training(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 定义评估函数
def evaluate_model(model, loader):
    model.eval()
    correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            predictions = model(imgs)
            _, predicted_classes = torch.max(predictions, 1)
            total_samples += lbls.size(0)
            correct_preds += (predicted_classes == lbls).sum().item()
    return correct_preds / total_samples

# 模型训练过程
print("Training ResNet18 model...")
net = create_model()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

tb_writer = SummaryWriter("logs/resnet18_model")
for epoch in range(3):  # 训练3个epoch
    avg_train_loss = run_training(net, train_loader, loss_function, optimizer)
    accuracy = evaluate_model(net, test_loader)
    print(f"Epoch [{epoch+1}/3] - Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")
    tb_writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    tb_writer.add_scalar('Accuracy/Test', accuracy, epoch)
tb_writer.close()
