import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sem.sem import legslbndm, lepoly, legslbdiff
from time import time

N = 15
m = 1



def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N+3):
		lepolys[i] = lepoly(i, x)
	return lepolys

def gen_phi(N, x, lepolys):
	phi = {}
	for l in range(N):
		phi[l] = lepolys[l] - lepolys[l+2]
	return phi

def gen_phi_x(N, x, lepolysx):
	phi = {}
	for l in range(N):
		phi[l] = lepolysx[l] - lepolysx[l+2]
	return phi

def gen_diff_lepoly(N, n, x, lepolys):
	lepolysx = np.zeros((N, 1))
	for i in range(n):
		if ((i+n) % 2) != 0:
			lepolysx += (2*i+1)*lepolys[i]
	return lepolysx

def gen_lepolysx(N, x, lepolys):
	lepolysx = {}
	for i in range(N+3):
		lepolysx[i] = gen_diff_lepoly(N+1, i, x, lepolys).reshape(N+1, 1)
	return lepolysx

def f2D(x,y):
	# m = np.random.normal(0, 1, 2)
	# w = (np.pi)*np.random.normal(0, 1, 4)
	m = [1, 1]
	w = 4*[1]
	return  m[0]*np.cos(w[0]*x + w[1]*y) + m[1]*np.sin(w[2]*x + w[3]*y)

#-----------------------------------------------------------------------
# Torch Funcs

def gen_lepolys_torch(N, x):
	lepolys = {}
	for i in range(N+3):
		if type(lepoly(i, x)) != np.ndarray:
			lepolys[i] = lepoly(i, x)
		else:
			lepolys[i] = torch.from_numpy(lepoly(i, x))
	return lepolys

def gen_phi_torch(N, x, lepolys):
	phi = {}
	for l in range(N):
		phi[l] = lepolys[l] - lepolys[l+2]
	return phi

def gen_phi_x_torch(N, x, lepolysx):
	phi = {}
	for l in range(N):
		phi[l] = lepolysx[l] - lepolysx[l+2]
	return phi

def gen_diff_lepoly_torch(N, n, x, lepolys):
	lepolysx = torch.zeros((N, 1))
	for i in range(n):
		if ((i+n) % 2) != 0:
			lepolysx += (2*i+1)*lepolys[i]
	return lepolysx

def gen_lepolysx_torch(N, x, lepolys):
	lepolysx = {}
	for i in range(N+3):
		lepolysx[i] = gen_diff_lepoly_torch(N+1, i, x, lepolys).reshape(N+1, 1)
	return lepolysx

def kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def f2D_torch(x,y):
	# m = np.random.normal(0, 1, 2)
	# w = (np.pi)*np.random.normal(0, 1, 4)
	m = [1, 1]
	w = 4*[1]
	return  m[0]*np.cos(w[0]*x + w[1]*y) + m[1]*np.sin(w[2]*x + w[3]*y)

#-------------------------------------------------------------------------------------------------
# 3D Plotting

def plot(x, y, z):
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.show()
	# exit()


if __name__ == '__main__':
	start = time()
	x = legslbndm(N+1)
	y = x.copy()
	D = legslbdiff(N+1, x)
	D2 = D@D
	D2 = D2[1:-1, 1:-1]
	lepolys = gen_lepolys(N, x)
	lepolysx = gen_lepolysx(N, x, lepolys)
	I = np.eye(N-1)
	L = np.kron(I, D2) + np.kron(D2, I)
	x1, y1 = np.meshgrid(x, x)
	f = f2D(x1, y1)
	f_ = f[1:-1, 1:-1]
	f_ = f_.ravel()
	u = np.linalg.solve(-L, f_)
	xx, yy = np.meshgrid(x[1:-1], x[1:-1])
	uu = np.zeros_like(x1)
	uu[1:-1, 1:-1] = np.reshape(u, (x.shape[0]-2, x.shape[0]-2))
	# plot(x1, y1, uu)
	ux = np.zeros_like(x)
	uy = ux.copy()
	u_x = np.zeros_like(uu)
	u_y = u_x.copy()
	fx = np.zeros_like(x)
	phi = gen_phi(N, x, lepolys)
	phi_x = gen_phi_x(N, x, lepolysx)
	lhs, rhs = m**2*[0], m**2*[0]
	denom = lepolys[N]**2
	for l in range(m):
		for j in range(m):
			phi1 = phi[l]
			phi1_x = phi_x[l]
			phi2 = phi[j]
			phi2_y = phi_x[j]
			for i in range(N+1):
				u_x[i:i+1, :] = (D@(uu[i:i+1, :].T)).T
				u_y[:, i:i+1] = D@uu[:, i:i+1]
				ux[i] = np.sum((u_x[i:i+1, :].T)*phi1_x*2/(N*(N+1))/denom)
				uy[i] = np.sum(u_y[:, i:i+1]*phi2_y*2/(N*(N+1))/denom)
				fx[i] = np.sum((f[i:i+1, :].T)*phi1*2/(N*(N+1))/denom)

			lhs[m*l+j] = np.sum(ux*phi2*2/(N*(N+1))/denom) + np.sum(uy*phi1*2/(N*(N+1))/denom)
			rhs[m*l+j] = np.sum(fx*phi2*2/(N*(N+1))/denom)

	lhs = np.array(lhs)
	rhs = np.array(rhs)
	err = lhs - rhs
	print("Numpy:", np.sum(err))
	num = time()-start
	print("dt:", num)
	#-------------------------------------------------------------------------------------------------------

	# Need: phi, phi_x, D, u (interior)
	#-------------------------------------------------------------------------------------------------------
	start = time()
	x = torch.from_numpy(legslbndm(N+1))
	y = x.clone()
	D = legslbdiff(N+1, x)
	D2 = D@D
	D2 = D2[1:-1, 1:-1]
	lepolys = gen_lepolys_torch(N, x)
	lepolysx = gen_lepolysx_torch(N, x, lepolys)
	I = torch.eye(N-1)
	# L = torch.einsum('ij,jk->ik', I, D2) + torch.einsum('ij,jk->ik', I, D2)
	L = kron(I, D2) + kron(D2, I)
	x1, y1 = np.meshgrid(x, x)
	f = torch.from_numpy(f2D_torch(x1, y1))
	f_ = f[1:-1, 1:-1]
	f_ = f_.reshape(f_.shape[0]**2, 1)
	# print(L.shape, type(L))
	# print(f_.shape, type(f_))
	u = torch.mm(-torch.inverse(L), f_)
	xx, yy = np.meshgrid(x[1:-1], x[1:-1])
	uu = torch.zeros_like(torch.from_numpy(x1))
	uu[1:-1, 1:-1] = torch.reshape(u, (x.shape[0]-2, x.shape[0]-2))
	# plot(x1, y1, uu.numpy())
	ux = torch.zeros_like(x)
	uy = ux.clone()
	u_x = torch.zeros_like(uu)
	u_y = u_x.clone()
	fx = torch.zeros_like(x)
	phi = gen_phi_torch(N, x, lepolys)
	phi_x = gen_phi_x_torch(N, x, lepolysx)
	lhs, rhs = m**2*[0], m**2*[0]
	denom = lepolys[N]**2
	for l in range(m):
		for j in range(m):
			phi1 = phi[l]
			phi1_x = phi_x[l]
			phi2 = phi[j]
			phi2_y = phi_x[j]
			for i in range(N+1):
				u_x[i:i+1, :] = (D@(uu[i:i+1, :].T)).T
				u_y[:, i:i+1] = D@uu[:, i:i+1]
				ux[i] = torch.sum((u_x[i:i+1, :].T)*phi1_x*2/(N*(N+1))/denom)
				uy[i] = torch.sum(u_y[:, i:i+1]*phi2_y*2/(N*(N+1))/denom)
				fx[i] = torch.sum((f[i:i+1, :].T)*phi1*2/(N*(N+1))/denom)

			lhs[m*l+j] = torch.sum(ux*phi2*2/(N*(N+1))/denom) + torch.sum(uy*phi1*2/(N*(N+1))/denom)
			rhs[m*l+j] = torch.sum(fx*phi2*2/(N*(N+1))/denom)

	lhs = torch.tensor(lhs)
	rhs = torch.tensor(rhs)
	err = lhs - rhs
	print("Torch:", torch.sum(err))
	tor = time()-start
	print("dt:", tor)
	print("\n", tor/num)