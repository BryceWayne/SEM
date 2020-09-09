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

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Burgers', choices=['Standard', 'Burgers', 'Helmholtz', 'BurgersT'])
parser.add_argument("--size", type=int, default=1) # BEFORE N
parser.add_argument("--N", type=int, default=31, choices=[int(2**i-1) for i in [4, 5, 6, 7, 8]]) 
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--rand_eps", type=bool, default=False)
args = parser.parse_args()


EQUATION = args.equation
SIZE = args.size
N = args.N
EPSILON = args.eps
EPS_FLAG = args.rand_eps
KIND = args.kind
SD = args.sd

def save_obj(data, name, equation, kind):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data', equation, kind)
	if os.path.isdir(path) == False:
		os.makedirs(f'data/{equation}/{kind}')
	with open(f'data/{equation}/{kind}/'+ name + '.pkl', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def create(N:int, epsilon:float):
	x, u, f, a = lg.lg_1d_standard(N, epsilon)
	x = x.reshape(1,x.shape[0])
	u = u.reshape(1,u.shape[0])
	f = f.reshape(1,f.shape[0])
	a = a.reshape(1,a.shape[0])
	return x, u, f, a


def create_fast(N:int, epsilon:float, size:int, eps_flag=False, equation='Standard', sd=1):
	def func(x: np.ndarray, equation: str, sd: float) -> np.ndarray:
		# Random force: mean=0, sd=1
		if equation == 'Burgers':
			m = np.random.normal(0, sd, 4)
			m = 2 + np.random.rand(2)
			n = np.pi*(1+2*np.random.rand(2))
			# mean = 0, sd=0.25
			# y = np.sin(2*pi*x)
			f = m[0]*np.sin(n[0]*x) + m[1]*np.cos(n[1]*x)
			m = np.array([m[0], m[1], n[0], n[1]])
		else:
			m = np.random.randn(4)
			f = m[0]*np.sin(m[1]*np.pi*x) + m[2]*np.cos(m[3]*np.pi*x)
		return f, m

	def gen_lepolys(N, x):
		lepolys = {}
		for i in range(N+3):
			lepolys[i] = sem.lepoly(i, x)
		return lepolys
		
	def generate(x, D, a, b, lepolys, epsilon, equation, sd):
		f, params = func(x, equation, sd)
		s_diag = np.zeros((N-1,1))
		if equation == 'Standard':
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

		elif equation == 'Burgers':
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

		elif equation == 'BurgersT':
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
			
		elif equation == 'Helmholtz':
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

	def loop(N, epsilon, size, lepolys, eps_flag, equation, a, b):
		if eps_flag == True:
			epsilons = np.random.uniform(1E0, 1E-6, SIZE)
		data = []
		U, F, ALPHAS, PARAMS = [], [], [], []
		for n in tqdm(range(size)):
			if eps_flag == True:
				epsilon = epsilons[n]
			if equation == 'BurgersT':
				u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd)
				for i, u_ in enumerate(u):
					if i < len(u):
						data.append([u[i], f[i], alphas[i], params, epsilon])
			elif equation == 'Burgers':
				u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd)
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
					u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd)
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
				u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd)
			data.append([u, f, alphas, params, epsilon])
		return data


	x = sem.legslbndm(N+1)
	D = sem.legslbdiff(N+1, x)
	lepolys = gen_lepolys(N, x)
	if equation == 'Helmholtz':
		a, b = np.zeros((N+1,)), np.zeros((N+1,))
		for i in range(1, N+2):
			k = i-1
			b[k] = -k*(k+1)/((k+2)*(k+3))
	else:
		a, b = 0, -1
	return loop(N, epsilon, size, lepolys, eps_flag, equation, a, b)


data = create_fast(N, EPSILON, SIZE, EPS_FLAG, EQUATION, SD)
data = np.array(data, dtype=object)

if EQUATION == 'Burgers':
	save_obj(data, f'{SIZE}N{N}sd{SD}', EQUATION, KIND)
else:
	save_obj(data, f'{SIZE}N{N}', EQUATION, KIND)