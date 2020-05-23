import torch
from torch import nn


# target output size of 5
m = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
input = torch.randn(1, 32, 64)
output = m(input)
print(output.shape)