import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=7, padding=3, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class NetU(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3) :
        super(NetU,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv2 = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fc1 = nn.Linear(filters*(self.d_out), self.d_out, bias=True)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = self.conv2(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], self.d_out)
        return out


class NetA(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3) :
        super(NetA,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out - 2
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv2 = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fc1 = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = self.conv2(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], self.d_out)
        return out