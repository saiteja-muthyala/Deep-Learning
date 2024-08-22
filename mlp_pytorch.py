#mlp using pytorch and compare the accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = CIFAR10(root = './data',train=True,download=True, transform=transform)
test_data = CIFAR10(root = './data',train=False,download=True, transform=transform)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

class Model_0(nn.Module):
    def __init__(self):
        super(Model_0, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128,10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = Model_0()
c = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for image,label in train_loader:
        optimizer.zero_grad()
        predictions = model(image)
        loss = c(predictions,label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss for Epoch {epoch}:{total_loss / len(train_loader)}")