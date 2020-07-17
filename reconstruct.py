#reconstruct.py
import torch
import numpy as np
from sem.sem import legslbndm, lepoly, legslbdiff
# import pdb

# Check if CUDA is available and then use it.
def get_device():
	if torch.cuda.is_available():  
		dev = "cuda:0" 
	else:  
		dev = "cpu"
	return torch.device(dev)

device = get_device()


def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N):
		lepolys[i] = lepoly(i, x)
	return lepolys


def basis(N, lepolys, equation):
	phi = torch.zeros((N-2,N))
	if equation in ('Standard', 'Burgers'):
		for i in range(N-2):
			phi[i,:] = torch.from_numpy(lepolys[i] - lepolys[i+2]).reshape(1,N)
	elif equation == 'Helmholtz':
		for i in range(N-2):
			coeff = -i*(i+1)/((i+2)*(i+3))
			phi[i,:] = torch.from_numpy(lepolys[i] + coeff*lepolys[i+2]).reshape(1,N)
	return phi.to(device)


def dx(N, x, lepolys):
	def gen_diff_lepoly(N, n, x,lepolys):
		lepoly_x = np.zeros((N, 1))
		for i in range(n):
			if ((i+n) % 2) != 0:
				lepoly_x += (2*i+1)*lepolys[i]
		return lepoly_x
	Dx = {}
	for i in range(N):
		Dx[i] = gen_diff_lepoly(N, i, x, lepolys).reshape(1, N)
	return Dx


def basis_x(N, phi, Dx, equation):
	phi_x = phi.clone()
	if equation in ('Standard', 'Burgers'):
		for i in range(N-2):
			phi_x[i,:] = torch.from_numpy(Dx[i] - Dx[i+2])
	elif equation == 'Helmholtz':
		for i in range(N-2):
			coeff = -i*(i+1)/((i+2)*(i+3))
			phi_x[i,:] = torch.from_numpy(Dx[i] + coeff*Dx[i+2])
	return phi_x.to(device)


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


def basis_xx(N, phi, Dxx, equation):
	phi_xx = phi.clone()
	if equation in ('Standard', 'Burgers'):
		for i in range(N-2):
			phi_xx[i,:] = torch.from_numpy(Dxx[i] - Dxx[i+2])
	elif equation == 'Helmholtz':
		for i in range(N-2):
			coeff = -i*(i+1)/((i+2)*(i+3))
			phi_xx[i,:] = torch.from_numpy(Dxx[i] + coeff*Dxx[i+2])
	return phi_xx.to(device)


def basis_vectors(N, equation):
	xx = legslbndm(N)
	lepolys = gen_lepolys(N, xx)
	lepoly_x = dx(N, xx, lepolys)
	lepoly_xx = dxx(N, xx, lepolys)
	phi = basis(N, lepolys, equation)
	phi_x = basis_x(N, phi, lepoly_x, equation)
	phi_xx = basis_xx(N, phi_x, lepoly_xx, equation)
	return xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx


def reconstruct(alphas, phi):
	B, i, j = alphas.shape
	P = torch.zeros((B, j, j+2), requires_grad=False).to(device)
	P[:,:,:] = phi
	T = torch.bmm(alphas,P)
	return T


def ODE2(eps, u, alphas, phi_x, phi_xx, equation):
	ux = reconstruct(alphas, phi_x)
	uxx = reconstruct(alphas, phi_xx)
	if equation == 'Standard':
		DE = -eps*uxx - ux
	elif equation == 'Burgers':
		DE = -eps*uxx + u*ux
	elif equation == 'Helmholtz':
		ku = 3.5
		DE = uxx + ku*u
	return DE


def weak_form1(eps, N, f, u, alphas, lepolys, phi, phi_x):
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	u_x = reconstruct(alphas, phi_x)
	LHS = eps*torch.sum(torch.square(u_x)*2/(N*(N+1))/denom, axis=2)
	RHS = torch.sum(f*u*2/(N*(N+1))/denom, axis=2)
	return LHS, RHS


def weak_form2(eps, N, f, u, alphas, lepolys, phi, phi_x, equation, nbfuncs):
	B, i, j = u.shape
	N -= 1
	LHS = torch.zeros((B, nbfuncs, 1)).to(device).float()
	RHS = torch.zeros((B, nbfuncs, 1)).to(device).float()
	phi = torch.transpose(phi, 0, 1)
	denom = torch.square(torch.from_numpy(lepolys[N]).to(device).float())
	denom = torch.transpose(denom, 0, 1)
	diffusion = 6*eps*alphas[:,:,0] # int(u_x*phi_0) = 6 *eps*a
	if equation == 'Standard':
		u_x = reconstruct(alphas, phi_x)
		ux_phi = u_x*phi[:,0]
		#SCALAR
		# convection = torch.sum(ux_phi*2/(N*(N+1))/denom, axis=2)
		# LHS = diffusion - convection
		# RHS = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2)
		#VECTOR
		convection = torch.sum(ux_phi*2/(N*(N+1))/denom, axis=2)
		LHS[:,0] = diffusion - convection
		RHS[:,0] = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2)
		if nbfuncs > 1:
			for i in range(1, nbfuncs):
				diffusion = -eps*(4*i+6)*(-1)*alphas[:,:,i]
				ux_phi = u_x*phi[:,i]
				# SCALAR
				# convection = torch.sum(ux_phi*2/(N*(N+1))/denom, axis=2)
				# LHS += diffusion - convection
				# RHS += torch.sum(2*f*phi[:,i]/(N*(N+1))/denom, axis=2)
				# VECTOR
				convection = torch.sum(ux_phi*2/(N*(N+1))/denom, axis=2)
				LHS[:,i] = diffusion - convection
				RHS[:,i] = torch.sum(2*f*phi[:,i]/(N*(N+1))/denom, axis=2)			
	elif equation == 'Burgers':
		phi_x = torch.transpose(phi_x, 0, 1)
		convection = torch.sum(0.5*u**2*phi_x[:,0]/(N*(N+1))/denom, axis=2)
		LHS[:,0] = diffusion - convection
		RHS[:,0] = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2)
		if nbfuncs > 1:
			for i in range(1, nbfuncs):
				diffusion = -eps*(4*i+6)*(-1)*alphas[:,:,i]
				convection = torch.sum(0.5*u**2*phi_x[:,i]/(N*(N+1))/denom, axis=2)
				LHS[:,i] = diffusion - convection
				RHS[:,i] = torch.sum(2*f*phi[:,i]/(N*(N+1))/denom, axis=2)
	elif equation == 'Helmholtz':
		ku = 3.5
		x = legslbndm(N+1)
		D_ = torch.from_numpy(legslbdiff(N+1, x)).to(device).float()
		D = torch.zeros((B, N+1, N+1)).to(device).float()
		D[:,:,:] = D_
		phi_x = torch.transpose(phi_x, 0, 1)
		u_ = torch.transpose(u, 1, 2)
		temp = torch.bmm(D,u_)
		temp = torch.transpose(temp, 1, 2)
		diffusion = torch.sum(2*temp*phi_x[:,0]/(N*(N+1))/denom, axis=2)
		reaction = ku*torch.sum(2*u*phi[:,0]/(N*(N+1))/denom, axis=2)
		LHS[:,0] = -diffusion + reaction 
		RHS[:,0] = torch.sum(2*f*phi[:,0]/(N*(N+1))/denom, axis=2)
		if nbfuncs > 1:
			for i in range(1, nbfuncs):
				diffusion = torch.sum(2*temp*phi_x[:,i]/(N*(N+1))/denom, axis=2)
				reaction = ku*torch.sum(2*u*phi[:,i]/(N*(N+1))/denom, axis=2)
				LHS[:,i] = -diffusion + reaction 
				RHS[:,i] = torch.sum(2*f*phi[:,i]/(N*(N+1))/denom, axis=2)
	return LHS, RHS
