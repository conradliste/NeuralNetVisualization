import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from extract_layers import extract_layers


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
layer_dict = extract_layers(net, (1, 28, 28), device=torch.device("cpu"))
for key in layer_dict:
    try:
        print(layer_dict[key]["bias"].shape)
    except KeyError:
        pass