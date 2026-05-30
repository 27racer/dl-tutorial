import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# 1. LOAD DATA
# ============================================================
# ToTensor() converts images to tensors and scales pixels to [0,1].
# Normalize subtracts the mean and divides by std (MNIST's known values),
# which helps the network train faster and more stably.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Downloads MNIST automatically the first time you run this.
train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# DataLoader feeds the model data in mini-batches.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)

# ============================================================
# 2. DEFINE THE MODEL
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # An image is 28x28 = 784 pixels. We flatten it into a vector
        # and pass it through fully-connected (linear) layers.
        self.fc1 = nn.Linear(784, 128)   # 784 inputs -> 128 hidden units
        self.fc2 = nn.Linear(128, 64)    # 128 -> 64
        self.fc3 = nn.Linear(64, 10)     # 64 -> 10 (one score per digit)

    def forward(self, x):
        x = x.view(-1, 784)        # flatten: (batch, 1, 28, 28) -> (batch, 784)
        x = F.relu(self.fc1(x))    # ReLU adds non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x)            # raw scores (logits) for the 10 classes
        return x

# ============================================================
# 3. TRAIN
# ============================================================
device = 'mps'
model = Net().to(device)

# CrossEntropyLoss is the standard loss for classification.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):                 # 5 passes over the data is plenty
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward passf
        loss = criterion(outputs, labels)
        loss.backward()                # compute gradients
        optimizer.step()               # update weights

    print(f"Epoch {epoch+1} done, last batch loss = {loss.item():.4f}")

# ============================================================
# 4. EVALUATE
# ============================================================
model.eval()
correct = 0
total = 0
with torch.no_grad():                  # no gradients needed for testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)   # pick highest-scoring digit
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {100 * correct / total:.2f}%")