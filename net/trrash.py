import torch
from torch import nn
from tqdm import tqdm

# # target output size of 5
# BATCH = 10
# input = torch.randn(BATCH, 64)
# basis = torch.randn(BATCH, 64)
# output = torch.bmm(input, basis)
# temp = torch.zeros(BATCH, 64, 1)
# temp2 = torch.randn(64, 1)
# for i in tqdm(range(BATCH)):
# 	temp[i,:,:] = temp2
# print(f"\n{temp.shape}")
# print(output.shape)
# print(output[:,0,0])

# output = torch.bmm(input, temp)
# print(temp2.T)
# print(temp[5,:,:].T)

# i, j = temp2.shape
# print(i, j)
# print(type(i))

A = torch.randn(1,30)
B = torch.randn(30,32)
print(A)
print(torch.mm(A,B))
temp = torch.mm(A,B).reshape(32,)
C = torch.zeros(10, 32)
for i in tqdm(range(10)):
	C[i,:] = temp
# A = A.reshape(10, 32)
print(C)