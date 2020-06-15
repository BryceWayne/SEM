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

"""
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   1819/1    0.007    0.000    6.201    6.201 {built-in method builtins.exec}
        1    0.001    0.001    6.201    6.201 training_a.py:2(<module>)
  1493/10    0.012    0.000    2.650    0.265 <frozen importlib._bootstrap>:978(_find_and_load)
  1487/10    0.006    0.000    2.650    0.265 <frozen importlib._bootstrap>:948(_find_and_load_unlocked)
  1438/11    0.006    0.000    2.647    0.241 <frozen importlib._bootstrap>:663(_load_unlocked)
  1316/11    0.004    0.000    2.647    0.241 <frozen importlib._bootstrap_external>:722(exec_module)
  1839/11    0.001    0.000    2.642    0.240 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
      101    0.001    0.000    1.858    0.018 __init__.py:1(<module>)
       45    1.530    0.034    1.688    0.038 {method 'to' of 'torch._C._TensorBase' objects}
        1    0.000    0.000    1.681    1.681 reconstruct.py:22(basis)
"""
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
	LHS = torch.zeros((u.shape[0],), requires_grad=False).to(device).float()
	RHS = torch.zeros((u.shape[0],), requires_grad=False).to(device).float()
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	u_x = reconstruct(alphas, phi_x)
	for index in range(u.shape[0]):
		LHS[index] = eps*torch.sum(torch.square(u_x[index,:])*2/(N*(N+1))/denom)
		RHS[index] = torch.sum(f[index,:]*u[index,:]*2/(N*(N+1))/denom)
	return LHS, RHS


def weak_form2(eps, N, f, u, alphas, lepolys, phi, phi_x):
	B, i, j = u.shape
	LHS = torch.zeros_like(u).to(device).float()
	RHS = torch.zeros_like(u).to(device).float()
	u_x = reconstruct(alphas, phi_x)
	phi = torch.transpose(phi, 1, 2)
	dummy = torch.zeros((B,i,j), requires_grad=False).to(device).float()
	for _ in range(B):
		dummy[_,:,:] = phi
	temp_sum = torch.bmm(u_x,dummy).reshape(B, phi.shape[1])
	denom = torch.square(torch.from_numpy(lepolys[N-1]).to(device).float())
	denom = torch.transpose(denom, 0, 1)
	LHS, RHS = 0, 0
	for l in range(B):
		temp_sum = 0
		difussion = -eps*(4*l+6)*(-1)*alphas[l,:]
		convection = torch.sum(temp_sum[l,:]*2/(N*(N+1))/denom)
		rhs = torch.sum(f*phi[l,:]*2/(N*(N+1))/denom)
		LHS += diffusion - convection
		RHS += rhs
	return LHS, RHS
