import numpy as np
import pickle
import os
import LG_1d as lg
from sem import sem as sem
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Standard')
parser.add_argument("--size", type=int, default=10)
parser.add_argument("--N", type=int, default=31)
parser.add_argument("--eps", type=float, default=1E-1)
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--rand_eps", type=bool, default=False)
args = parser.parse_args()


EQUATION = args.equation
SIZE = args.size
N = args.N
EPSILON = args.eps
EPS_FLAG = args.rand_eps
KIND = args.kind

def save_obj(data, name, equation, kind):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data', equation, kind)
	if os.path.isdir(path) == False:
		os.makedirs(f'data/{equation}/{kind}')
	with open(f'data/{equation}/{kind}/'+ name + '.pkl', 'wb') as f:
		pickle.dump(data	, f, pickle.HIGHEST_PROTOCOL)


def create(N:int, epsilon:float):
	x, u, f, a = lg.lg_1d_standard(N, epsilon)
	x = x.reshape(1,x.shape[0])
	u = u.reshape(1,u.shape[0])
	f = f.reshape(1,f.shape[0])
	a = a.reshape(1,a.shape[0])
	return x, u, f, a


def create_fast(N:int, epsilon:float, size:int, eps_flag=False, equation='Standard'):
	def func(x: np.ndarray) -> np.ndarray:
		# Random force: mean=0, sd=1
		m = np.random.randn(4)
		f = 0.5*m[0]*np.sin(m[1]*np.pi*x) + 0.5*m[2]*np.cos(m[3]*np.pi*x)
		return f, m

	def gen_lepolys(N, x):
		lepolys = {}
		for i in range(N+3):
			lepolys[i] = sem.lepoly(i, x)
		return lepolys

	def generate(x, D, a, b, lepolys, epsilon, equation):
		f, params = func(x)
		if equation == 'Standard':
			s_diag = np.zeros((N-1,1))
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

		elif equation == 'Burgers':
			s_diag = np.zeros((N-1,1))
			for ii in range(1, N):
				k = ii - 1
				s_diag[k] = -(4*k+6)*b
			S = s_diag*np.eye(N-1)
			Mass = epsilon*S
			error, tolerance, u_old, force = 1, 1E-14, 0*f.copy(), f.copy()
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
			return u_sol, f, alphas, params
			
		elif equation == 'Helmholtz':
			ku = 3.5
			s_diag = np.zeros((N-1,1))
			M = np.zeros((N-1,N-1))
			for ii in range(1, N):
				k = ii - 1
				s_diag[k] = -(4*k+6)*b[k]
				phi_k_M = lepolys[k] + a[ii]*lepolys[k+1] + b[ii]*lepolys[k+2]
				for jj in range(1,N):
					if np.abs(ii-jj) <= 2:
						l = jj-1
						psi_l_M = lepolys[l] + a[jj]*lepolys[l+1] + b[jj]*lepolys[l+2]
						M[jj-1,ii-1] = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepolys[N]**2))

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
				u += alphas[i_ind]*(lepolys[i_ind] + a[i_ind]*lepolys[i_ind] + b[i_ind]*lepolys[i_ind])
			return u, f, alphas, params

	def loop(N, epsilon, size, lepolys, eps_flag, equation):
		if eps_flag == True:
			epsilons = np.random.uniform(1E0, 1E-6, SIZE)
		data = []
		U, F, ALPHAS, PARAMS = [], [], [], []
		for n in tqdm(range(size)):
			if eps_flag == True:
				epsilon = epsilons[n]
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation)
			data.append([u, f, alphas, params, epsilon])
		return data


	x = sem.legslbndm(N+1)
	D = sem.legslbdiff(N+1, x)
	lepolys = gen_lepolys(N, x)
	if equation == 'Helmholtz':
		a, b = np.zeros((N+1, 1)), np.zeros((N+1, 1))
		for i in range(1, N+2):
			k = i-1
			b[k] = -k*(k+1)/((k+2)*(k+3))
	else:
		a, b = 0, -1
	print(f"EQUATION: {EQUATION}")
	return loop(N, epsilon, size, lepolys, eps_flag, equation)


data = create_fast(N, EPSILON, SIZE, EPS_FLAG, EQUATION)
data = np.array(data)

save_obj(data, f'{SIZE}N{N}', EQUATION, KIND)
