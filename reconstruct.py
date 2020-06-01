#training.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sem.sem import legslbndm, lepoly, legslbdiff


# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N):
		lepolys[i] = lepoly(i, x)
	return lepolys

def diff(N, T, D):
	x = legslbndm(N+1)
	# D = torch.from_numpy(legslbdiff(N+1, x)).to(device).float()
	T_ = T.clone()
	for i in range(T.shape[0]):
		element = torch.mm(D,T[i,:].reshape(T.shape[1], 1)).reshape(T.shape[1],)
		T_[i,:] = element
	T = T_.clone()
	del T_
	return T


def reconstruct(N, alphas, lepolys):
	i, j = alphas.shape
	j += 2
	M = torch.zeros((j-2,j), requires_grad=False).to(device)
	T = torch.zeros((i, j), requires_grad=False).to(device)
	for jj in range(1, j-1):
		i_ind = jj - 1
		element = torch.from_numpy(lepolys[i_ind] - lepolys[i_ind+2]).reshape(j,)
		M[i_ind,:] = element
	for ii in range(i):
		a = alphas[ii,:].detach().reshape(1, j-2)
		sol = torch.mm(a,M).reshape(j,)
		T[ii,:] = sol
	return T


def ODE(N, eps, u, M):
	ux = diff(N, u, M)
	uxx = diff(N, ux, M)
	return -eps*uxx - ux


def ODE2(N, eps, u, alphas, lepolys, D):
	i, j = alphas.shape
	j += 2
	M = torch.zeros((j-2,j), requires_grad=False).to(device)
	T = torch.zeros((i, j), requires_grad=False).to(device)
	for jj in range(1, j-1):
		i_ind = jj - 1
		element = torch.from_numpy(lepolys[i_ind] - lepolys[i_ind+2]).to(device).float()#.reshape(j,)
		d1 = torch.mm(D,element)
		d2 = torch.mm(D,d1)
		M[i_ind,:] = torch.transpose(eps*d2 + d1, 0, 1)

	for ii in range(i):
		a = alphas[ii,:].detach().reshape(1, j-2)
		sol = torch.mm(a,M).reshape(j,)
		T[ii,:] = sol
	return T
