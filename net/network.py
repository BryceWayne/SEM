import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=True) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class Net(nn.Module) :
	def __init__(self, D_in, Filters, D_out) :
		super(Net,self).__init__()
		self.conv1 = conv1d(1, 1)
		self.conv2 = conv1d(1, 1)
		self.fc = nn.Linear(64, D_out)

	def forward(self, x) :
		out = self.conv1(x)
		out = F.relu(out)
		out = self.conv2(out)
		out = F.relu(out)
		out = self.conv2(out)
		out = F.relu(out)
		out = self.fc(out)
		return out

