import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1d(in_planes, out_planes, stride=1, bias=False, kernel_size=5) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)

class Net(nn.Module) :
	def __init__(self, d_in, filters, d_out) :
		super(Net,self).__init__()
		self.conv = conv1d(d_in, filters)
		self.conv1 = conv1d(filters, filters)
		self.fc1 = nn.Linear(filters*d_out, d_out, bias=True)
		self.fc1.weight = nn.init.xavier_uniform_(self.fc1.weight)
		self.d_in = d_in
		self.d_out = d_out

	def forward(self, x) :
		out = self.conv(x)
		out = F.relu(out)
		out = self.conv1(out)
		out = F.relu(out)
		out = self.conv1(out)
		out = out.view(-1, 2048)
		out = self.fc1(out)
		out = out.view(out.shape[0], 1, out.shape[1])
		return out

