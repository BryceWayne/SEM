import numpy as np
import pickle
import os
import LG_1d as lg
from sem import sem as sem
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from mpl_toolkits import mplot3d 
from reconstruct import dx
from multiprocessing import Pool # TODO	

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Standard2D', choices=['Standard', 'Standard2D', 'Burgers', 'Helmholtz', 'BurgersT'])
parser.add_argument("--size", type=int, default=1) # BEFORE N
parser.add_argument("--N", type=int, default=63, choices=[int(2**i-1) for i in [4, 5, 6, 7, 8]]) 
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--forcing", type=str, default='uniform', choices=['normal', 'uniform'])
parser.add_argument("--rand_eps", type=bool, default=False)
args = parser.parse_args()


equation = args.equation
size = args.size
N = args.N
epsilon = args.eps
eps_flag = args.rand_eps
kind = args.kind
sd = args.sd
forcing = args.forcing


def save_obj(data, name, equation, kind):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data', equation, kind)
	if os.path.isdir(path) == False:
		os.makedirs(f'data/{equation}/{kind}')
	with open(f'data/{equation}/{kind}/'+ name + '.pkl', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N+3):
		lepolys[i] = sem.lepoly(i, x)
	return lepolys

def gen_lepolysx(N, x, D):
	lepolysx = {}
	for i in range(N+3):
		lepolysx[i] = D@sem.lepoly(i, x)
	return lepolysx


def func(x, equation, sd, forcing):
	if forcing == 'uniform':
		m = 3 + 2*np.random.rand(2)
		n = np.pi*(1+2*np.random.rand(2))
		f = m[0]*np.sin(n[0]*x) + m[1]*np.cos(n[1]*x)
		m = np.array([m[0], m[1], n[0], n[1]])
	elif forcing == 'normal':
		m = np.random.normal(0, sd, 4)
		f = m[0]*np.sin(m[1]*np.pi*x) + m[2]*np.cos(m[3]*np.pi*x)
	return f, m


def func2D(x, y, equation, sd, forcing):
	if forcing == 'uniform':
		m = np.random.rand(2)
		w = np.random.rand(4)*(np.pi/2)
	elif forcing == 'normal':
		m = np.random.normal(0, sd, 2)
		w = np.random.normal(0, sd, 4)*(np.pi/2)

	w *= 2
	f = m[0]*np.cos(w[0]*x + w[1]*y) + m[1]*np.sin(w[2]*x + w[3]*y)
	m = np.array([m[0], m[1], w[0], w[1], w[2], w[3]])
	return f, m


def standard(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag):
	M = np.zeros((N-1,N-1))
	for ii in range(1, N):
		k = ii - 1
		s_diag[ii-1] = -(4*k+6)*b
		phi_k_M = D@(lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2])
		for jj in range(1,N):
			if np.abs(ii-jj) <=2:
				l = jj-1
				psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
				M[jj-1,ii-1] = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepolys[N]**2))

	S = s_diag*np.eye(N-1)
	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i - 1
		g[i-1] = (2*k+1)/(N*(N+1))*np.sum(f*(lepolys[k])/(lepolys[N]**2))
	g[N-1] = 1/(N+1)*np.sum(f/lepolys[N])

	bar_f = np.zeros((N-1,))
	for i in range(1,N):
		k = i-1
		bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

	Mass = epsilon*S-M
	u = np.linalg.solve(Mass, bar_f)
	alphas = np.copy(u)
	g[0], g[1] = u[0], u[1] + a*u[0]

	for i in range(3, N):
		k = i - 1
		g[i-1] = u[i-1] + a*u[i-2] + b*u[i-3]

	g[N-1] = a*u[N-2] + b*u[N-3]
	g[N] = b*u[N-2]
	u = np.zeros((N+1,))
	for i in range(1,N+2):
		_ = 0
		for j in range(1, N+2):
			k = j-1
			L = lepolys[k]
			_ += g[j-1]*L[i-1]
		u[i-1] = _[0]
	return u, f, alphas, params


def standard2D(x, D, a, b, lepolys, lepolysx, epsilon, equation, sd, forcing, f, params, s_diag):
	D2 = D**2
	D2 = D2[1:-1, 1:-1]
	I = np.eye(x.shape[0]-2)
	L = np.kron(I, D2) + np.kron(D2, I)
	f_ = f[1:-1, 1:-1]
	u = np.linalg.solve(-L, f_.ravel())
	x1, y1 = np.meshgrid(x, x)
	xx, yy = np.meshgrid(x[1:-1], x[1:-1])
	uu = np.zeros_like(x1)

	uu[1:-1, 1:-1] = np.reshape(u, (x.shape[0]-2, x.shape[0]-2))
	ux = np.zeros_like(x)
	uy = ux.copy()
	u_x = np.zeros_like(uu)
	u_y = u_x.copy()
	fx = np.zeros_like(x)

	lhs, rhs = 10*[0], 10*[0]
	m = 1
	for l in range(m+1):
	    for j in range(m+1):
	        
	        phi1 = lepolys[l] - lepolys[l+2]
	        phi1_x = lepolysx[l] - lepolysx[l+2]
	        
	        phi2 = lepolys[j] - lepolys[j+2]
	        phi2_x = lepolysx[j] - lepolysx[j+2]
	        
	        
	        for i in range(N+1):
	            u_x[i, :] = D@uu[i, :].T
	            u_y[:, i] = D@uu[:, i]
	            
	            ux[i] = np.sum(u_x[i, :].T*(phi1_x)*2/(N*(N+1))/(lepolys[N]**2))
	            uy[i] = np.sum(u_y[:, i]*(phi2_x)*2/(N*(N+1))/(lepolys[N]**2))
	            
	            fx[i] = np.sum(f[i,:].T*(phi1)*2/(N*(N+1))/(lepolys[N]**2))
	        
	        
	        lhs[(m+1)*l+j] = np.sum(ux.T*phi2*2/(N*(N+1))/(lepolys[N]**2)) + np.sum(uy.T*phi1*2/(N*(N+1))/(lepolys[N]**2))
	        rhs[(m+1)*l+j] = np.sum(fx.T*phi2*2/(N*(N+1))/(lepolys[N]**2))

	lhs = np.array(lhs)
	rhs = np.array(rhs)
	err = lhs - rhs
	# print(err)
	alphas = 0
	return u, f, alphas, params


def burgers(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag):
	for ii in range(1, N):
		k = ii - 1
		s_diag[k] = -(4*k+6)*b
	S = s_diag*np.eye(N-1)
	Mass = epsilon*S
	error, tolerance, u_old, force = 1, 1E-9, 0*f.copy(), f.copy()
	iterations = 0
	while error > tolerance:
		f_ = force - u_old*(D@u_old)
		g = np.zeros((N+1,))
		for i in range(1,N+1):
			k = i-1
			g[k] = (2*k+1)/(N*(N+1))*np.sum(f_*(lepolys[k])/(lepolys[N]**2))
		g[N-1] = 1/(N+1)*np.sum(f_/lepolys[N])

		bar_f = np.zeros((N-1,))
		for i in range(1,N):
			k = i-1
			bar_f[k] = g[k]/(k+1/2) + a*g[k+1]/(k+3/2) + b*g[k+2]/(k+5/2)

		alphas = np.linalg.solve(Mass, bar_f)
		u_sol = np.zeros((N+1, 1))
		for ij in range(1, N):
			i_ind = ij - 1
			u_sol += alphas[i_ind]*(lepolys[i_ind] + a*lepolys[i_ind+1] + b*lepolys[i_ind+2])

		error = np.max(u_sol - u_old)
		u_old = u_sol
		iterations += 1
	u = u_sol
	return u, f, alphas, params


def burgersT(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag):
	M = np.zeros((N-1, N-1))
	tol, T, dt = 1E-9, 5E-4, 1E-4
	t_f = int(T/dt)
	u_pre, u_ans, f_ans, alphas_ans = np.sin(np.pi*x), [], [], []
	for ii in range(1, N):
		k = ii - 1
		s_diag[k] = -(4*k + 6)*b
		phi_k_M = lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2]
		for jj in range(1, N):
			if np.abs(ii-jj) <= 2:
				l = jj-1
				psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
				entry = psi_l_M*phi_k_M*2/(N*(N+1))/(lepolys[N]**2)
				M[l, k] = np.sum(entry)

	S = s_diag*np.eye(N-1)
	Mass = epsilon*S + (1/dt)*M
	
	for t_idx in np.linspace(1, t_f, t_f, endpoint=True):
		error, tolerance, u_old, force = 1, tol, u_pre, np.cos(t_idx*dt)*f

		iterations = 0
		while error > tolerance:
			f_ = force - u_old*(D@u_old) + (1/dt)*u_pre
			g = np.zeros((N+1,))
			for i in range(1,N+1):
				k = i-1
				g[k] = (2*k+1)/(N*(N+1))*np.sum(f_*(lepolys[k])/(lepolys[N]**2))
			g[N-1] = 1/(N+1)*np.sum(f_/lepolys[N])

			bar_f = np.zeros((N-1,))
			for i in range(1,N):
				k = i-1
				bar_f[k] = g[k]/(k+1/2) + a*g[k+1]/(k+3/2) + b*g[k+2]/(k+5/2)

			alphas = np.linalg.solve(Mass, bar_f)
			u_sol = np.zeros((N+1, 1))
			for ij in range(1, N):
				i_ind = ij - 1
				u_sol += alphas[i_ind]*(lepolys[i_ind] + a*lepolys[i_ind+1] + b*lepolys[i_ind+2])

			error = np.max(u_sol - u_old)
			u_old = u_sol.copy()
			iterations += 1

		u_ans.append(u_sol)
		f_ans.append(force)
		alphas_ans.append(alphas)
		u_pre = u_sol
	u, f, alphas = u_ans, f_ans, alphas_ans
	return u, f, alphas, params


def helmholtz(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag):
	ku = 3.5
	M = np.zeros((N-1, N-1))
	for ii in range(1, N):
		k = ii - 1
		s_diag[k] = -(4*k + 6)*b[k]
		phi_k_M = lepolys[k] + a[k]*lepolys[k+1] + b[k]*lepolys[k+2]
		for jj in range(1, N):
			if np.abs(ii-jj) <= 2:
				l = jj-1
				psi_l_M = lepolys[l] + a[l]*lepolys[l+1] + b[l]*lepolys[l+2]
				entry = psi_l_M*phi_k_M*2/(N*(N+1))/(lepolys[N]**2)
				M[l, k] = np.sum(entry)

	S = s_diag*np.eye(N-1)
	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i-1
		g[k] = (2*k+1)/(N*(N+1))*np.sum(f*(lepolys[k])/(lepolys[N]**2))
	g[N] = 1/(N+1)*np.sum(f/lepolys[N])

	bar_f = np.zeros((N-1,))
	for i in range(1,N):
		k = i-1
		bar_f[k] = g[k]/(k+1/2) + a[k]*g[k+1]/(k+3/2) + b[k]*g[k+2]/(k+5/2)

	Mass = -S + ku*M
	alphas = np.linalg.solve(Mass, bar_f)

	u = np.zeros((N+1, 1))
	for ij in range(1,N):
		i_ind = ij-1
		u += alphas[i_ind]*(lepolys[i_ind] + a[i_ind]*lepolys[i_ind+1] + b[i_ind]*lepolys[i_ind+2])
	return u, f, alphas, params


def generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing):
	if equation == 'Standard2D':
		x1, y1 = np.meshgrid(x, x)
		f, params = func2D(x1, y1, equation, sd, forcing)
	else:
		f, params = func(x, equation, sd, forcing)
	s_diag = np.zeros((N-1,1))
	if equation == 'Standard':
		u, f, alphas, params = standard(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag)
	elif equation == 'Standard2D':
		u, f, alphas, params = standard2D(x, D, a, b, lepolys, lepolysx, epsilon, equation, sd, forcing, f, params, s_diag)
		# plot3D(x, u)
	elif equation == 'Burgers':
		u, f, alphas, params = burgers(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag)
	elif equation == 'BurgersT':
		u, f, alphas, params = burgersT(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag)		
	elif equation == 'Helmholtz':
		u, f, alphas, params = helmholtz(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag)
	return u, f, alphas, params


def plot3D(x, u):
	# This import registers the 3D projection, but is otherwise unused.
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	
	u = np.reshape(u, (N-1, N-1))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(x, x)
	Z = 0*X.copy()
	Z[1:-1,1:-1] = u
	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	# ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


def create_fast(N, epsilon, size, eps_flag=False, equation='Standard', sd=1, forcing='uniform'):
	if equation == 'Helmholtz':
		a, b = np.zeros((N+1,)), np.zeros((N+1,))
		for i in range(1, N+2):
			k = i-1
			b[k] = -k*(k+1)/((k+2)*(k+3))
	else:
		a, b = 0, -1
	return loop(N, epsilon, size, lepolys, eps_flag, equation, a, b, forcing)


def loop(N, epsilon, size, lepolys, eps_flag, equation, a, b, forcing):
	if eps_flag == True:
		epsilons = np.random.uniform(1E0, 1E-6, size)
	data = []
	U, F, ALPHAS, PARAMS = [], [], [], []
	for n in tqdm(range(size)):
		if eps_flag == True:
			epsilon = epsilons[n]
		if equation == 'Standard':
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
		elif equation == 'Standard2D':
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
		elif equation == 'BurgersT':
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
			for i, u_ in enumerate(u):
				if i < len(u):
					data.append([u[i], f[i], alphas[i], params, epsilon])
		elif equation == 'Burgers':
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
			LHS, RHS = 0, 0
			for _ in range(10):
				phi_0 = lepolys[_] - lepolys[_+2]
				phi_x = D@phi_0
				diffusion = -epsilon*(4*_+6)*(-1)*alphas[_]
				denom = lepolys[N]**2
				convection = np.sum(u**2*phi_x/(N*(N+1))/denom)
				LHS += diffusion - convection
				RHS += np.sum(2*f*phi_0/(N*(N+1))/denom)
			while np.abs(LHS-RHS) > 1E-5 and np.linalg.norm(u, ord=2) < 1E-2:
				u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
				LHS, RHS = 0, 0
				for _ in range(10):
					phi_0 = lepolys[_] - lepolys[_+2]
					phi_x = D@phi_0
					diffusion = -epsilon*(4*_+6)*(-1)*alphas[_]
					denom = lepolys[N]**2
					convection = np.sum(u**2*phi_x/(N*(N+1))/denom)
					LHS += diffusion - convection
					RHS += np.sum(2*f*phi_0/(N*(N+1))/denom)
		else:
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
		data.append([u, f, alphas, params, epsilon])
	return data


x = sem.legslbndm(N+1)
D = sem.legslbdiff(N+1, x)
lepolys = gen_lepolys(N, x)
if equation == 'Standard2D':
	lepolysx = gen_lepolysx(N, x, D)
# print(equation)

data = create_fast(N, epsilon, size, eps_flag, equation, sd, forcing)
data = np.array(data, dtype=object)

if forcing == 'normal':
	save_obj(data, f'{size}N{N}sd{sd}', equation, kind)
elif forcing == 'uniform':
	save_obj(data, f'{size}N{N}uniform', equation, kind)

