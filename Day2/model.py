import torch
from torch import nn

class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    chen = Chen()
    x = torch.ones((64, 3, 32, 32))
    y = chen(x)
    print(y.shape)
