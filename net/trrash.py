import torch
from torch import nn

a = torch.randn(64, 1, 64)
a = a.permute(0,2,1)
print("Input:", a.shape)  
m = nn.Conv1d(64, 32, 5, padding=2) 
out = m(a)
print("Conv1 out:", out.size())
n = nn.Conv1d(32, 32, 5, padding=2) 
out = n(out)
print("Conv2 out:", out.size())
print("conv1d_1:", m)
print("conv1d_2:", n)