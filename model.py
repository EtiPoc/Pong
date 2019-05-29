import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.linear_1 = nn.Linear(3136, 784)
        self.linear_2 = nn.Linear(784, n_actions)

    def forward(self, x):
        x = x/255
        x = F.relu(self.bn1(self.conv_1(x)))
        x = F.relu(self.pool_1(self.conv_2(x)))
        x = F.relu(self.bn2(self.conv_3(x)))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        x = self.linear_2(x)
        return x

    def save(self, path):
        torch.save(path)

    def load(self, path):
        torch.load(path)
