#training.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sem.sem import legslbndm, lepoly


def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N):
		lepolys[i] = lepoly(i, x)
	return lepolys

def reconstruct(N, alphas, lepolys):
	i, j = alphas.shape
	j += 2
	T = torch.zeros((i, j))
	T = T.numpy()
	temp = alphas.clone().to('cpu').detach().numpy()
	for ii in range(i):
		a = temp[ii,:].reshape(j-2, 1)
		sol = np.zeros((j,1))
		for jj in range(1,j-1):
			i_ind = jj - 1
			sol += a[i_ind]*(lepolys[i_ind]-lepolys[i_ind+2])
		T[ii,:] = sol.T[0]
	return T
