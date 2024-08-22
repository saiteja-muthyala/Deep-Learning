#cnn using pytorch and compare the accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = CIFAR10(root = './data',train=True,download=True, transform=transform)
test_data = CIFAR10(root = './data',train=False,download=True, transform=transform)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

  
class Cifar_cnn(nn.Module):
    def __init__(self):
        super(Cifar_cnn, self).__init__()
        self.convo1 = nn.Conv2d(3, 6, 5)
        self.convo2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.convo1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.convo2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = x.view(-1,5*5*16)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
model = Cifar_cnn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    loss_epoch = 0
    for images,label in train_loader:
        pred = model(images)
        loss = criterion(pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch += loss.item()
    print(f"Loss for Epoch {epoch}:{loss_epoch / len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images,label in test_loader:
        pred = model.forward(images)
        _,predicted = torch.max(pred.data,1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
    print(f"Accuracy : {(correct/total)*100}")