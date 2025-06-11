import torch
import torch.nn.functional as F

input_data = torch.tensor([[1, 2, 0, 3, 1],
                           [0, 1, 2, 3, 1],
                           [1, 2, 1, 0, 0],
                           [5, 2, 3, 1, 1],
                           [2, 1, 0, 1, 1]])

kernel_data = torch.tensor([[1, 2, 1],
                            [0, 1, 0],
                            [2, 1, 0]])

# 显示原始尺寸
print(f"Input shape: {input_data.shape}")
print(f"Kernel shape: {kernel_data.shape}")

# 调整输入和卷积核的尺寸
input_data = input_data.unsqueeze(0).unsqueeze(0)  # 将尺寸从 (5, 5) 改为 (1, 1, 5, 5)
kernel_data = kernel_data.unsqueeze(0).unsqueeze(0)  # 将尺寸从 (3, 3) 改为 (1, 1, 3, 3)
print(f"Adjusted Input shape: {input_data.shape}")
print(f"Adjusted Kernel shape: {kernel_data.shape}")

# 卷积操作，步长为1
output_1 = F.conv2d(input_data, kernel_data, stride=1)
print(f"Output with stride 1:\n{output_1}")

# 卷积操作，步长为2
output_2 = F.conv2d(input_data, kernel_data, stride=2)
print(f"Output with stride 2:\n{output_2}")

# 填充为1，步长为1
output_3 = F.conv2d(input_data, kernel_data, stride=1, padding=1)
print(f"Output with padding 1:\n{output_3}")
