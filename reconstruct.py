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


def basis(lepolys):
	L = max(list(lepolys.keys()))
	phi = torch.zeros((L+1,L+1))
	for i in range(L-1):
		phi[i] = torch.from_numpy(lepolys[i] - lepolys[i+2]).reshape(1,L+1)
	return phi.to(device)


def basis_x(phi, Dx):
	phi_x = phi.clone()
	for i in range(phi.shape[0]-2):
		phi_x[i] = torch.from_numpy(Dx[i] - Dx[i+2])
	return phi_x.to(device)


def basis_xx(phi, Dxx):
	phi_xx = phi.clone()
	for i in range(phi.shape[0]-2):
		phi_xx[i] = torch.from_numpy(Dxx[i] - Dxx[i+2])
	return phi_xx.to(device)


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
	T_ = T.clone()
	for i in range(T.shape[0]):
		element = torch.mm(D,T[i,:].reshape(T.shape[1], 1)).reshape(T.shape[1],)
		T_[i,:] = element
	T = T_.clone()
	del T_
	return T


def reconstruct(N, alphas, phi):
	i, j = alphas.shape
	j += 2
	M = torch.zeros((j-2,j), requires_grad=False).to(device)
	T = torch.zeros((i, j), requires_grad=False).to(device)
	for jj in range(1, j-1):
		i_ind = jj - 1
		M[i_ind,:] = phi[i_ind].reshape(j,)
	for ii in range(i):
		a = alphas[ii,:].detach().reshape(1, j-2)
		T[ii,:] = torch.mm(a,M).reshape(j,)
	del M
	return T


def ODE2(eps, u, alphas, phi_x, phi_xx):
	i, j = alphas.shape
	j += 2
	M = torch.zeros((j-2,j), requires_grad=False).to(device)
	T = torch.zeros((i, j), requires_grad=False).to(device)
	for jj in range(1, j-1):
		i_ind = jj - 1
		M[i_ind,:] = -eps*phi_xx[i_ind] - phi_x[i_ind]

	for ii in range(i):
		a = alphas[ii,:].detach().reshape(1, j-2)
		sol = torch.mm(a,M).reshape(j,)
		T[ii,:] = sol
	del M
	return T


def weak_form1(eps, N, f, u, alphas, lepolys, Dx):
	LHS = torch.zeros((u.shape[0],), requires_grad=False).to(device).float()
	RHS = torch.zeros((u.shape[0],), requires_grad=False).to(device).float()
	for index in range(u.shape[0]):
		u_x = torch.zeros((N,1), requires_grad=False).to(device).float()
		a = alphas[index,:]
		for i in range(N-2):
			diff1 = torch.from_numpy(Dx[i].reshape(Dx[i].shape[1],1)).to(device).float()
			u_x += a[i]*diff1
		LHS[index] = eps*torch.sum(u_x*u_x*2/(N*(N+1))/(torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())))
		RHS[index] = torch.sum(f*u*2/(N*(N+1))/(torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())))
	return LHS, RHS


def weak_form2(eps, N, f, u, alphas, lepolys, DX):
	def wf2(eps, N, f, u, alphas, lepolys, DX):
		cumulative_error = 0
		for l in range(5):
			temp_sum = 0
			difussion = -eps*(4*l+6)*(-1)*u[l]
			for k in range(N-1):
				phi_k_M = u[k]*DX[k].reshape(DX.shape[1],)
				temp_sum = temp_sum + phi_k_M*(lepolys[l] - lepolys[l+2])
			convection = np.sum(temp_sum*2/(N*(N+1))/(np.square(lepolys[N-1])))
			rhs = np.sum(f*(lepolys[l] - lepolys[l+2])*2/(N*(N+1))/(np.square(lepolys[N-1])))
			cumulative_error = cumulative_error + diffusion - convection - rhs
		return cumulative_error

	return LHS, RHS