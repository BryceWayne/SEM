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


def dx(N, x, lepolys):
	def gen_diff_lepoly(N, n, x,lepolys):
		lepoly_x = np.zeros((N,1))
		for i in range(n):
			if ((i+n) % 2) != 0:
				lepoly_x += (2*i+1)*lepolys[i]
		return lepoly_x
	Dx = {}
	for i in range(N):
		Dx[i] = gen_diff_lepoly(N, i, x, lepolys).reshape(1, N)
	return Dx


def dxx(N, x, lepolys):
	def gen_diff2_lepoly(N, n, x,lepolys):
		lepoly_xx = np.zeros((N,1))
		for i in range(n-1):
			if ((i+n) % 2) == 0:
				lepoly_xx += (i+1/2)*(n*(n+1)-i*(i+1))*lepolys[i]
		return lepoly_xx
	Dxx = {}
	for i in range(N):
		Dxx[i] = gen_diff2_lepoly(N, i, x, lepolys).reshape(1, N)
	return Dxx


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
		a = alphas[ii,:].reshape(1, j-2)
		sol = torch.mm(a,M).reshape(j,)
		T[ii,:] = sol
	del M, element, sol
	return T


def ODE(eps, u, Dx, Dxx):
	u = u.reshape(u.shape[0], u.shape[1], 1)
	ux = torch.zeros_like(u)
	uxx = torch.zeros_like(u)
	for i in range(u.shape[0]):
		ux[i,:] = torch.mm(Dx,u[i,:, :])
	del Dx
	for i in range(u.shape[0]):
		uxx[i,:] = torch.mm(Dxx,u[i,:, :])
	del Dxx
	return (-eps*uxx - ux).reshape(u.shape[0], u.shape[1])


def ODE2(eps, u, alphas, lepolys, DX, DXX):
	i, j = alphas.shape
	j += 2
	M = torch.zeros((j-2,j), requires_grad=False).to(device)
	T = torch.zeros((i, j), requires_grad=False).to(device)
	for jj in range(1, j-1):
		i_ind = jj - 1
		d1 = torch.from_numpy(DX[i_ind] - DX[i_ind+2]).to(device).float()
		d2 = torch.from_numpy(DXX[i_ind] - DXX[i_ind+2]).to(device).float()
		de = -eps*d2 - d1
		M[i_ind,:] = de

	for ii in range(i):
		a = alphas[ii,:].reshape(1, j-2)
		sol = torch.mm(a,M).reshape(j,)
		T[ii,:] = sol
	return T
