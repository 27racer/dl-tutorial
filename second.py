import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# 1. LOAD DATA 
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)

# ============================================================
# 2. DEFINE THE MODEL 
# ============================================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size)
        # Layer 1: 1 input channel (grayscale) -> 32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Layer 2: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling halves the image size each time (28 -> 14 -> 7)
        self.pool = nn.MaxPool2d(2, 2)

        # After two poolings: 64 maps of size 7x7 = 64*7*7 = 3136 values
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout randomly "switches off" neurons during training
        # to prevent over-reliance on any one of them (reduces overfitting)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x starts as (batch, 1, 28, 28) -- NOTE: no flattening!
        x = self.pool(F.relu(self.conv1(x)))   # -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7)             # NOW flatten for the linear layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # raw scores for the 10 digits
        return x

# ============================================================
# 3. TRAIN  (only the model name changed: CNN() instead of Net())
# ============================================================
device = 'mps'
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done, last batch loss = {loss.item():.4f}")

# ============================================================
# 4. EVALUATE 
# ============================================================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {100 * correct / total:.2f}%")