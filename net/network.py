import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class ResNet(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2):
        super(ResNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.conv = conv1d(d_in, filters, kernel_size=kernel_size, padding=padding)
        # self.n1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1d(filters, filters, kernel_size=kernel_size, padding=padding)
        # self.n2 = nn.BatchNorm1d(filters)
        self.conv2 = conv1d(filters, filters, kernel_size=kernel_size, padding=padding)
        self.residual = nn.Sequential(
            self.relu,
            self.conv1,
            self.relu,
            self.conv2)
        self.fc1 = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        x = self.conv(x)
        out = self.residual(x)
        x = F.relu(x + out)
        out = self.residual(x)
        x = F.relu(x + out)
        out = self.residual(x)
        x = F.relu(x + out)
        out = self.residual(x)
        x = F.relu(x + out)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = x.view(x.shape[0], self.d_out)
        return x

    # def initialize(self):
    #     nn.init.zeros_(self.n1.weight)


class NetU(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3) :
        super(NetU,self).__init__()
        self.d_in = d_in # (batch, 1, 32)
        self.filters = filters # (batch, 32, 32)
        self.d_out = d_out # (batch, 32)
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv2 = conv1d(filters, 2*filters, kernel_size=self.kern, padding=self.pad)
        self.conv3 = conv1d(2*filters, 3*filters, kernel_size=self.kern, padding=self.pad)
        self.conv4 = conv1d(3*filters, 4*filters, kernel_size=self.kern, padding=self.pad)
        self.conv5 = conv1d(4*filters, 5*filters, kernel_size=self.kern, padding=self.pad)
        self.fc1 = nn.Linear(5*filters*(self.d_out), self.d_out, bias=True)
    def forward(self, x):
        # CHECK FULLY CONVOLUTIONAL NETWORK
        # SPATIAL TRANSFORM NETWORK (DeepMind 2015/16)
        # DEFORMABLE CONVOLUTION NETWORK ()
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2(out))
        out = self.conv2(out)
        out = out.flatten(start_dim=1)
        # global pooling to avg feature maps
        out = self.fc1(out)
        out = out.view(out.shape[0], self.d_out)
        return out


class NetA(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3) :
        super(NetA,self).__init__()
        self.d_in = d_in
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv2 = conv1d(filters, 2*filters, kernel_size=self.kern, padding=self.pad)
        self.conv3 = conv1d(2*filters, 3*filters, kernel_size=self.kern, padding=self.pad)
        self.conv4 = conv1d(3*filters, 4*filters, kernel_size=self.kern, padding=self.pad)
        self.conv5 = conv1d(4*filters, 5*filters, kernel_size=self.kern, padding=self.pad)
        self.fc1 = nn.Linear(5*filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], self.d_out)
        return out