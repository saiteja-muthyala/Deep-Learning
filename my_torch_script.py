import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
x = torch.randn(100,10)
y = torch.randn(100,1)

#build the Arch
class  mst_Samplenet(nn.Module):
    def __init__(self):
        super(mst_Samplenet, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
model = mst_Samplenet()
c = nn.MSELoss()
opt = optim.SGD(model.parameters(),lr = 0.001)   
l =[]
for epochs in range(500):
    pred = model.forward(x)
    loss = c(pred,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    l.append(loss.item())
    print(f"loss- {epochs+1}:{loss}")

#PLOT
plt.plot(l)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()