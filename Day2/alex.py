import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 128, 5, padding=2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.Conv2d(192, 128, 3, padding=1),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    model = AlexNet()
    input_data = torch.randn(1, 3, 224, 224)
    output = model(input_data)
    print("Output shape:", output.shape)
