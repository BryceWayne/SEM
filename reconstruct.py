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


def basis_vectors(D_out):
	xx = legslbndm(D_out)
	lepolys = gen_lepolys(D_out, xx)
	lepoly_x = dx(D_out, xx, lepolys)
	lepoly_xx = dxx(D_out, xx, lepolys)
	phi = basis(D_out, lepolys)
	phi_x = basis_x(D_out, phi, lepoly_x)
	phi_xx = basis_xx(D_out, phi_x, lepoly_xx)
	return xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx


def reconstruct(alphas, phi):
	B, i, j = alphas.shape
	P = torch.zeros((B, j, j+2), requires_grad=False).to(device)
	P[:,:,:] = phi
	T = torch.bmm(alphas,P)
	return T


def ODE2(eps, u, alphas, phi_x, phi_xx, equation='Standard'):
	if equation == 'Standard':
		DE = reconstruct(alphas, -eps*phi_xx - phi_x)
	elif equation == 'Burgers':
		DE = reconstruct(alphas, -eps*phi_xx + u*phi_x)
	return DE


def weak_form1(eps, N, f, u, alphas, lepolys, phi, phi_x):
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	u_x = reconstruct(alphas, phi_x)
	LHS = eps*torch.sum(torch.square(u_x)*2/(N*(N+1))/denom)
	RHS = torch.sum(f*u*2/(N*(N+1))/denom)
	return LHS, RHS


def weak_form2(eps, N, f, u, alphas, lepolys, phi, phi_x, equation='Standard'):
	B, i, j = u.shape
	N -= 1
	phi = torch.transpose(phi, 0, 1)
	denom = torch.square(torch.from_numpy(lepolys[N]).to(device).float())
	denom = torch.transpose(denom, 0, 1)
	diffusion = 6*eps*alphas[:,:,0]
	if equation == 'Standard':
		u_x = reconstruct(alphas, phi_x)
		ux_phi = u_x*phi[:,0] #sum(phi)
		convection = torch.sum(ux_phi*2/(N*(N+1))/denom, axis=2)
		LHS = diffusion - convection
		RHS = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2) #f*sum(phi)
	elif equation == 'Burgers':
		phi_x = torch.transpose(phi_x, 0, 1)
		convection = torch.sum(u**2*phi_x[:,0]/(N*(N+1))/denom, axis=2)
		LHS = torch.abs(diffusion - convection) #  
		RHS = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2)
	return LHS, RHS
