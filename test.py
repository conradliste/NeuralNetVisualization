import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from torchvision import datasets
import torchvision.transforms as transforms
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
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# L
net = Net()
viz_net = NetVisual(net, [784], device=torch.device("cpu"))
viz = nnVisual(viz_net)
layers_dict = viz_net.layers_dict
#viz.render()
#open_file("./media/videos/1080p60/nnVisual.mp4")


## Specify loss and optimization functions
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 30  # suggest training between 20-50 epochs

net.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        data_flat = torch.flatten(data, start_dim=1)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data_flat)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data_flat.size(0)
        viz_net.update_layers_dict(data_flat)
        
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    print(net.parameters())
    print(viz_net.layers_dict["Linear-0"]["weights"])
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))

