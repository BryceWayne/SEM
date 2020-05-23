import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)

def norm(planes, norm_type="b") :
    if norm_type == "g" :
        return nn.GroupNorm(num_groups=min(32,planes), num_channels=planes)
    elif norm_type == "g2" :
        return nn.GroupNorm(num_groups=int(planes/2), num_channels=planes)
    elif norm_type == "g4" :
        return nn.GroupNorm(num_groups=int(planes/4), num_channels=planes)
    elif norm_type == "g8" :
        return nn.GroupNorm(num_groups=int(planes/8), num_channels=planes)
    elif norm_type == "l" :
        return nn.GroupNorm(num_groups=1, num_channels=planes)
    elif norm_type == "i" :
        return nn.GroupNorm(num_groups=planes, num_channels=planes)
    return nn.BatchNorm1d(planes)

class ResBlock(nn.Module) :
    expansion = 1

    def __init__(self, in_planes, out_planes, coef=1, stride=1, downsample=None, norm_type="b", **kwargs) :
        super(ResBlock,self).__init__()
        self.n1 = norm(in_planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1d(in_planes, 32)
        self.n2 = norm(32, norm_type)
        self.conv2 = conv1d(32, 32)
        self.coef = coef
        self.residual = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )


    def forward(self, x) :
        shortcut = x
        out = self.coef * self.residual(x)
        return out + shortcut

    def initialize(self) :
        nn.init.zeros_(self.n1.weight)

class Net(nn.Module) :
    def __init__(self, d_in, filters, d_out) :
        super(Net,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out
        self.conv1 = conv1d(d_in, filters)
        self.resblock = ResBlock(filters, filters)
        self.avg = nn.AdaptiveAvgPool1d(self.d_out)
        self.fc1 = nn.Linear(2048, d_out, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        # out = self.pool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        # out = F.relu(out)
        # print(out.shape)
        out = out.view(-1, 32*64)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = out.view(out.shape[0], self.d_out)
        return out


class U(nn.Module) :
    def __init__(self, d_in, filters, d_out) :
        super(U,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out
        self.conv1 = conv1d(d_in, filters)
        self.conv2 = conv1d(filters, filters)
        self.conv3 = conv1d(filters, filters)
        self.resblock = ResBlock(filters, filters)
        self.pool = nn.MaxPool1d(5, padding=2)
        self.fc1 = nn.Linear(2048, d_out, bias=True)
        self.fc2 = nn.Linear(4*d_out, d_out, bias=True)

    def forward(self, x) :
        out = self.conv1(x)
        out = F.relu(out)
        # out = self.pool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        # out = F.relu(out)
        # print(out.shape)
        out = out.view(-1, 32*64)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = out.view(out.shape[0], self.d_out)
        return out

