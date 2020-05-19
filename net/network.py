import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)

class Net(nn.Module) :
	def __init__(self, d_in, filters, d_out) :
		super(Net,self).__init__()
		self.conv = conv1d(d_in, filters)
		self.conv1 = conv1d(filters, 2*filters)
		filters *= 2
		self.conv2 = conv1d(filters, 2*filters)
		self.fc1 = nn.Linear(2*filters**2, d_out)
		self.fc2 = nn.Linear(1024, d_out)
		self.d_in = d_out
		self.d_out = d_out

	def forward(self, x) :
		out = self.conv(x)
		out = F.relu(out)
		out = self.conv1(out)
		out = F.relu(out)
		out = self.conv2(out)
		out = out.view(100, -1)
		out = self.fc1(out)
		out = out.view(out.shape[0], 1, out.shape[1])
		return out

