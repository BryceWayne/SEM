import torch
from torch import nn


# target output size of 5
m = nn.AdaptiveMaxPool1d(64)
input = torch.randn(1, 64, 8)
output = m(input)
print(output.shape)