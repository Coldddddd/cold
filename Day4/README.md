# 《深度学习代码学习笔记》

## 一、代码 1：`train_alex.py`

### 功能概述
该代码实现了基于 AlexNet 架构的图像分类模型训练，采用自定义数据集进行训练。

### 代码解析

**数据集加载与预处理**
- 利用自定义的 `ImageTxtDataset` 类加载数据集，而非常见的 CIFAR-10 数据集。
- 数据集路径与格式：图像路径和标签信息保存在 `train.txt` 文件中，而图像是放在 `D:\dataset\image2\train` 文件夹下。
- 数据预处理步骤：
  - 通过 `transforms.Resize(224)` 将图像尺寸调整至 224x224，以契合 AlexNet 的输入规格。
  - 运用 `transforms.RandomHorizontalFlip()` 实现随机水平翻转，旨在丰富数据的多样性。
  - 借助 `transforms.ToTensor()` 把图像转化为张量格式。
  - 采用 `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` 对图像进行归一化操作，此处选用的均值和标准差是 ImageNet 数据集常用参数。

**模型架构**
- 定义了一个简化版的 AlexNet 模型：
  - 包含 5 个卷积层以及 3 个全连接层。
  - 卷积层运用 `MaxPool2d` 进行下采样操作。
  - 最后一层输出维度为 10，适用于 10 类别的分类任务。
  - 输入为 3 通道的 RGB 图像数据。

**训练流程**
- 运用 `DataLoader` 加载数据，设定批量大小为 64。
- 选用交叉熵损失函数（`CrossEntropyLoss`）和随机梯度下降优化器（`SGD`），学习率设置为 0.01，动量参数为 0.9。
- 每隔 500 步记录一次训练损失情况，并借助 TensorBoard 实现可视化展示。
- 每完成一个 epoch，对测试集进行一次评估，统计整体损失和准确率。
- 每个 epoch 结束后，将模型保存为 `.pth` 格式文件。

**测试流程**
- 测试阶段，利用 `torch.no_grad()` 禁用梯度计算，目的是降低内存占用并加速计算过程。
- 计算测试集的总损失和准确率，并将这些结果记录到 TensorBoard 中，便于后续的分析和查看。

### 学习要点
- 自定义数据集的运用：深入了解如何处理自定义格式的数据集，特别是通过文本文件加载图像路径和标签的方式，以及如何对数据进行预处理以满足模型的输入需求。
- AlexNet 架构的剖析：把握 AlexNet 的核心架构，包括卷积层、池化层、全连接层的功能及相互关系，同时学习根据实际任务需求对模型输出层进行合理调整。
- 训练与测试流程的掌握：熟悉 PyTorch 中模型训练和测试的完整流程，涵盖数据加载、损失计算、优化器更新、模型评估与保存等环节，同时学会运用 TensorBoard 进行训练过程的可视化呈现。
- 数据增强技术的应用：认识数据增强技术（如随机水平翻转）对于提升模型泛化能力的重要性，并学会将其应用到实际的模型训练过程中。

## 二、代码 2：`transformer.py`

### 功能介绍
该代码实现了一个基于 Transformer 架构的 Vision Transformer（ViT）模型，用于处理序列化的图像数据。

### 代码解析

**模块构成**
- **FeedForward 模块**：由一个线性层、GELU 激活函数、Dropout 以及另一个线性层组成，并通过 `LayerNorm` 进行归一化处理。
- **Attention 模块**：实现了多头自注意力机制，利用 `Softmax` 函数计算注意力权重，同时借助 `rearrange` 和 `repeat` 函数对张量形状进行灵活处理。
- **Transformer 模块**：由多个 Transformer 层构成，每层均包含一个注意力模块和一个前馈模块，且加入了残差连接（`x = attn(x) + x` 和 `x = ff(x) + x`）。
- **ViT 模型**：将输入图像序列化为 patches 后，通过 Transformer 模型进行处理，同时引入位置嵌入（`pos_embedding`）和类别嵌入（`cls_token`），最终经全连接层输出分类结果。

**模型架构**
- 输入为序列化的图像（`time_series`），其形状为 `(batch_size, channels, seq_len)`。
- 模型将输入序列划分为多个大小为 `patch_size` 的 patches。
- 通过 Transformer 模型对这些 patches 进行深度处理。
- 最终输出的分类结果形状为 `(batch_size, num_classes)`。

**测试示例**
- 创建 ViT 模型实例，向其输入一个随机生成的张量（`time_series`）。
- 输出的 logits 形状为 `(batch_size, num_classes)`，验证了模型能够正常运行并产生预期的输出结果。

### 学习要点
- Transformer 架构的解析：深入理解 Transformer 的核心结构，包括多头自注意力机制、前馈网络以及残差连接的作用和意义，同时明确 `LayerNorm` 和 `Dropout` 在 Transformer 中的用途及应用场景。
- Vision Transformer（ViT）的实现：掌握如何将 Transformer 架构应用于图像数据的处理，通过对图像进行序列化为 patches 的操作来适配 Transformer 的输入要求，同时理解位置嵌入和类别嵌入在模型中的关键作用，并学会利用 Transformer 模型完成图像分类任务。
- `einops` 库的操作：学习使用 `einops` 库简化复杂的张量操作，例如 `rearrange` 和 `repeat` 函数的灵活运用，提高代码的可读性和可维护性。
- 模型输入输出的把控：明确 ViT 模型的输入数据格式（序列化的图像）和输出数据格式（分类结果），以便更好地理解和应用该模型。

## 三、总结
1. `train_alex.py`：基于 AlexNet 的图像分类模型，着重学习了自定义数据集的处理、数据预处理技巧、模型训练与测试的完整流程。
2. `transformer.py`：基于 Transformer 的 Vision Transformer 模型，重点学习了 Transformer 架构的核心原理、Vision Transformer 的实现细节以及 `einops` 库的实际应用。

通过对这两个代码的深入学习，进一步加深了我对卷积神经网络（CNN）和 Transformer 架构的认识，同时也提升了我在处理自定义数据集和运用数据增强技术方面的能力。