import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv2d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class ResNet(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=5):
        super(ResNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=padding)
        self.n1 = nn.GroupNorm(1, self.filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        self.n2 = nn.GroupNorm(1, self.filters)
        self.conv2 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        self.residual = nn.Sequential(
            self.n1,
            self.relu,
            self.conv1,
            self.n2,
            self.relu,
            self.conv2)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x) #1
        if self.blocks != 0:
            for block in range(self.blocks):
                out = F.relu(out + self.residual(out))
        # out = self.n1(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class NetA(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetA,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = F.relu(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class NetB(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetB,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        m = nn.Sigmoid()
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class Net2D(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetB,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv2d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv2d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        m = nn.Sigmoid()
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], self.d_in, self.d_out)
        return out