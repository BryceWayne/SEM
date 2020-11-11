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

def weights_xavier(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)    

def init_optim(model):
    params = {
              'history_size': 10,
              'tolerance_grad': 1E-15,
              'tolerance_change': 1E-15,
              'max_eval': 10,
                }
    return torch.optim.LBFGS(model.parameters(), **params)


def swish(x):
    return x * torch.sigmoid(x)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class RelMSELoss(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.mse = nn.MSELoss()
        self.batch = batch
    def forward(self,yhat,y):
        loss = self.mse(yhat,y)/self.batch
        return loss


def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class Linear(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=0):
        super(ResNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class ResNet(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=0):
        super(ResNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=padding)
        # self.n1 = nn.GroupNorm(1, self.filters)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        # self.n2 = nn.GroupNorm(1, self.filters)
        # self.conv2 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        # self.residual = nn.Sequential(
        #     self.n1,
        #     self.relu,
        #     self.conv1,
        #     self.n2,
        #     self.relu,
        #     self.conv2)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class ResNetD(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=5):
        super(ResNetD, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=1)
        self.n1 = nn.GroupNorm(1, self.filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=0)
        self.n2 = nn.GroupNorm(1, self.filters)
        self.conv2 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=0)
        self.residual = nn.Sequential(
            self.n1,
            self.relu,
            self.conv1,
            self.n2,
            self.relu,
            self.conv2)
        # self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x) #1
        if self.blocks != 0:
            for block in range(self.blocks):
                out = self.relu(out + self.residual(out))
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
        # torch.nn.Dropout(0.2)
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


class NetC(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetC,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        m = self.swish
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class NetD(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=0, blocks=0, activation='swish'):
        super(NetD, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.activation = activation.lower()
        self.m = swish
        self.d_out = d_out
        self.swish = swish
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=1)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=0)
        self.dim = d_in*(d_out - 4*(self.blocks + 1))*filters
        self.fcH = nn.Linear(self.dim, self.d_out, bias=True)
    def forward(self, x):
        if self.activation == 'relu':
            m = self.relu
        elif self.activation == 'sigmoid':
            m = self.sigmoid
        elif self.activation == 'swish':
            m = self.swish

        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))

        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class FC(nn.Module):
    def __init__(self, d_in, hidden, d_out, layers=1, activation='relu') :
        super(FC, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.relu = nn.ReLU(inplace=True)
        self.swish = swish
        self.sigmoid = nn.Sigmoid
        self.hidden = hidden
        self.layers = layers
        self.activation = activation
        self.d_in = d_in
        self.d_out = d_out
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, d_out)
    def forward(self, x):
        if self.activation.lower() == 'relu':
            m = self.relu
        elif self.activation.lower() == 'sigmoid':
            m = self.sigmoid
        elif self.activation.lower() == 'swish':
            m = self.swish
        out = self.relu(self.layer2(x))
        for _ in range(self.layers):
            out = self.relu(self.layer2(out))
        out = self.layer3(out)
        out = out.view(out.shape[0], self.d_in, self.d_out)
        return out


def conv2d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class Net2D(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net2D,self).__init__()
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