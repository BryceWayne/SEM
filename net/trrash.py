import torch
from torch import nn


a = torch.randn(100, 1, 64)
print("Input:", a.shape)
m = nn.Conv1d(1, 32, 5, padding=2) 
out = m(a)
print("Conv1 out:", out.shape)
n = nn.Conv1d(32, 32, 5, padding=2) 
out = n(out)
print("Conv2 out:", out.shape)
out = n(out)
print("Conv3 out:", out.shape)
out = out.view(-1, 2048)
print("view:", out.shape)
o = nn.Linear(2048,1)
out = o(out)
print("FC out:", out.shape)
print("conv1d_1:", m)
print("conv1d_2:", n)