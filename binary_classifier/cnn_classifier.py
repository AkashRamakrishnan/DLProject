from torch import nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    def __init__(self, no_outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features= 90)
        self.fc2 = nn.Linear(in_features = 90, out_features = 45)
        self.out = nn.Linear(in_features= 45, out_features = no_outputs)
        
    def forward(self, tensor):
        # hidden layer 1
        tensor = self.conv1(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, kernel_size = 2, stride= 2)
        # hidden layer 2
        tensor = self.conv2(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, kernel_size = 2, stride = 2)
        # hidden layer 3
        tensor = tensor.reshape(-1, 12 * 4* 4)
        tensor = self.fc1(tensor)
        tensor = F.relu(tensor)
        # hidden layer 4
        tensor = self.fc2(tensor)
        tensor = F.relu(tensor)
        # output layer
        tensor = self.out(tensor)
        return tensor