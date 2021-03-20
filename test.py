import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from extract_layers import extract_layers
from visualize_net import nnVisual, NetVisual
from manim.utils.file_ops import open_file


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

net = Net()
viz_net = NetVisual(net, [256], device=torch.device("cpu"))
viz = nnVisual(viz_net.net_visual)
viz.render()
open_file("./media/videos/1080p60/nnVisual.mp4")
