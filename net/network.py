import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes) :
    return torch.nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class Net(nn.Module) :
    def __init__(self, init_channel=16, **kwargs) :
        super(Net,self).__init__()
        channel = init_channel
        self.conv1 = conv1d(1,channel)
        self.conv2 = conv1d(channel, channel)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv2(x)

    def loss(self) :
        return torch.nn.MSELoss()
