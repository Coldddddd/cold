import torch
from PIL import Image
from torchvision import transforms
from model import *

# 设置图片路径
img_path = "./Image/img.png"

# 加载图片并转换为RGB
img = Image.open(img_path).convert('RGB')
print(img)

# 进行图像预处理：调整大小并转换为tensor
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor()         # 转换为tensor
])

img_tensor = preprocess(img)
print(img_tensor.shape)

# 加载训练好的模型
model = torch.load("model_save\\chen_9.pth", weights_only=False).cpu()

# 调整图片tensor形状，准备输入模型
img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度
img_tensor = img_tensor.cpu()  # 确保使用CPU

# 设置模型为评估模式
model.eval()

# 在不计算梯度的情况下进行推理
with torch.no_grad():
    prediction = model(img_tensor)

# 输出预测的标签
print(prediction.argmax(dim=1))
