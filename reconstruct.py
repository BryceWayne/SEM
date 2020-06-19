#reconstruct.py
import torch
import numpy as np
from sem.sem import legslbndm, lepoly, legslbdiff
# import pdb

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

###
def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N):
		lepolys[i] = lepoly(i, x)
	return lepolys


def basis(N, lepolys):
	phi = torch.zeros((N-2,N))
	for i in range(N-2):
		phi[i,:] = torch.from_numpy(lepolys[i] - lepolys[i+2]).reshape(1,N)
	return phi.to(device)

###
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


def basis_x(N, phi, Dx):
	phi_x = phi.clone()
	for i in range(N-2):
		phi_x[i,:] = torch.from_numpy(Dx[i] - Dx[i+2])
	return phi_x.to(device)

###
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


def basis_xx(N, phi, Dxx):
	phi_xx = phi.clone()
	for i in range(N-2):
		phi_xx[i,:] = torch.from_numpy(Dxx[i] - Dxx[i+2])
	return phi_xx.to(device)


def reconstruct(alphas, phi):
	B, i, j = alphas.shape
	P = torch.zeros((B, j, j+2), requires_grad=False).to(device)
	P[:i,:,:] = phi
	T = torch.zeros((B, i, j+2), requires_grad=False).to(device)
	T = torch.bmm(alphas,P)
	return T


def ODE2(eps, u, alphas, phi_x, phi_xx):
	DE = reconstruct(alphas, -eps*phi_xx - phi_x)
	return DE


def weak_form1(eps, N, f, u, alphas, lepolys, phi_x):
	B = u.shape[0]
	# LHS = torch.zeros((B,)).to(device).float()
	# RHS = torch.zeros((B,)).to(device).float()
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	u_x = reconstruct(alphas, phi_x)
	LHS = eps*torch.sum(torch.square(u_x)*2/(N*(N+1))/denom)
	RHS = torch.sum(f*u*2/(N*(N+1))/denom)
	# for index in range(u.shape[0]):
	# 	LHS[index] = eps*torch.sum(torch.square(u_x[index,:])*2/(N*(N+1))/denom)
	# 	RHS[index] = torch.sum(f[index,:]*u[index,:]*2/(N*(N+1))/denom)
	return LHS, RHS


def weak_form2(eps, N, f, u, alphas, lepolys, phi, phi_x):
	B, i, j = u.shape
	# print("\n",B, i, j)
	LHS = torch.zeros_like(u).to(device).float()
	RHS = torch.zeros_like(u).to(device).float()
	u_x = reconstruct(alphas, phi_x)
	phi = torch.transpose(phi, 0, 1)
	dummy = torch.zeros((B,j,j-2), requires_grad=False).to(device).float()
	dummy[:,:,:] = phi
	ux_phi = torch.bmm(u_x,dummy).reshape(B, i, j-2)
	N -= 1
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	diff = torch.from_numpy(np.array([[-eps*(4*l+6)*(-1) for l in range(j-2)]])).to(device).float()
	diffusion = diff*alphas
	temp = torch.bmm(f,dummy)
	convection = ux_phi*2/(N*(N+1))/denom
	LHS = torch.sum(diffusion - convection, axis=2)
	RHS = torch.sum(temp*2/(N*(N+1))/denom, axis=2)
	# print(LHS.shape, RHS.shape)
	return LHS, RHS
