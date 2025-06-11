import time
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


class ModifiedAlexNet(nn.Module):
    def __init__(self):
        super(ModifiedAlexNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.fc_block(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = ModifiedAlexNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_steps = 0
test_steps = 0
epochs = 10
log_writer = SummaryWriter("../logs_train")

start_time = time.time()

for epoch in range(epochs):
    print(f"--- Training Epoch {epoch+1} ---")
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        predictions = model(inputs)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps += 1
        if train_steps % 500 == 0:
            print(f"Step {train_steps}, Training Loss: {loss.item()}")
            log_writer.add_scalar("Training Loss", loss.item(), train_steps)

    elapsed_time = time.time() - start_time
    print(f"Time taken for epoch {epoch+1}: {elapsed_time:.2f} seconds")

    model.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            total_test_loss += loss.item()

            correct_preds = (predictions.argmax(1) == labels).sum()
            total_correct += correct_preds

    print(f"Test Loss: {total_test_loss}")
    print(f"Test Accuracy: {total_correct / len(test_data):.4f}")
    
    log_writer.add_scalar("Test Loss", total_test_loss, test_steps)
    log_writer.add_scalar("Test Accuracy", total_correct / len(test_data), test_steps)
    test_steps += 1

    torch.save(model.state_dict(), f"saved_models/alexnet_epoch_{epoch+1}.pth")
    print("Model saved.")

log_writer.close()
