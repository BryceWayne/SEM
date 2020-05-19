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
		self.conv1 = conv1d(filters, filters)
		self.conv2 = conv1d(filters, 2*filters)
		self.fc = nn.Linear(1, 1)

	def forward(self, x) :
		out = self.conv(x)
		out = F.relu(out)
		out = self.conv1(out)
		out = F.relu(out)
		out = self.conv2(out)
		out = self.fc(out)
		return out

