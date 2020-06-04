import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=7, padding=3, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class Net(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3) :
        super(Net,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out - 2
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv2 = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        # self.conv2 = conv1d(filters, 2*filters, kernel_size=self.kern, padding=self.pad)
        # self.conv3 = conv1d(2*filters, 3*filters, kernel_size=self.kern, padding=self.pad)
        # self.conv4 = conv1d(3*filters, 4*filters, kernel_size=self.kern, padding=self.pad)
        # self.conv5 = conv1d(4*filters, 5*filters, kernel_size=self.kern, padding=self.pad)
        # self.resblock = ResBlock(filters, filters)
        self.fc1 = nn.Linear(filters*(self.d_out+2), self.d_out, bias=True)
        # self.fc2 = nn.Linear(5*filters*(self.d_out+2), (self.d_out+2)**2, bias=True)
    def forward(self, x):
        out = torch.tanh(self.conv1(x))
        out = torch.tanh(self.conv2(out))
        out = torch.tanh(self.conv2(out))
        out = torch.tanh(out)
        # out = F.relu(self.conv3(out))
        # out = F.relu(self.conv4(out))
        # out = self.conv5(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], self.d_out)
        return out

