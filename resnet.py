import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms,models
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.optim as optim

t = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = CIFAR10(root = './data',train=True,download=True, transform=t)
test_data = CIFAR10(root = './data',train=False,download=True, transform=t)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for params in model.parameters():
    params.requires_grad = False

in_features = model.fc.in_features    #fc  = Fully Connected Layer
model.fc = nn.Linear(in_features,10)    

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0
    for data,target in train_loader:
        optimizer.zero_grad()
        op = model(data)
        loss = criterion(op,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch : {epoch+1},Loss: {running_loss/len(train_loader)}")


model.eval()
correct,total = 0,0
with torch.no_grad():
    for images,label in test_loader:
        pred = model.forward(images)
        _,predicted = torch.max(pred.data,1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
    print(f"Accuracy : {(correct/total)*100}")

model = models.resnet18(pretrained = True)
for params in model.parameters():
    params.requires_grad = True

in_features = model.fc.in_features    #fc  = Fully Connected Layer
model.fc = nn.Linear(in_features,10)