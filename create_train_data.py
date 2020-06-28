import pickle
import os
import LG_1d as lg
from sem import sem as sem
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Standard')
parser.add_argument("--size", type=int, default=10)
parser.add_argument("--N", type=int, default=31)
parser.add_argument("--eps", type=float, default=1E-1)
parser.add_argument("--rand_eps", type=bool, default=False)
args = parser.parse_args()


EQUATION = args.equation
SIZE = args.size
N = args.N
EPSILON = args.eps
EPS_FLAG = args.rand_eps

def save_obj(obj, name, equation):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data', equation)
	if os.path.isdir(path) == False:
		os.makedirs(f'data/{equation}')
	with open(f'data/{equation}/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def create(N:int, epsilon:float):
	x, u, f, a = lg.lg_1d_standard(N, epsilon)
	x = x.reshape(1,x.shape[0])
	u = u.reshape(1,u.shape[0])
	f = f.reshape(1,f.shape[0])
	a = a.reshape(1,a.shape[0])
	return x, u, f, a


def create_fast(N:int, epsilon:float, size:int, eps_flag=False, equation='Standard'):
	def func(x: np.ndarray) -> np.ndarray:
		# Random force (0,2)
		m = 2*np.random.rand(4) - 1
		# m = np.random.rand(4)
		# m = np.array([0.75, 0.75, 0.75, 0.27])
		f = m[0]*np.sin(m[1]*np.pi*x) + m[2]*np.cos(m[3]*np.pi*x)
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
			error, tolerance, u_old, force = 1, 1E-12, 0*f, f.copy()
			iterations = 0
			# print(u_old.shape, force.shape, D.shape)
			while error > tolerance:
				f_ = force - u_old*(D@u_old)
				# print(f.shape)
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
				# print(alphas.shape, u_sol.shape)
				for ij in range(1, N):
					i_ind = ij - 1
					u_sol += alphas[i_ind]*(lepolys[i_ind] + a*lepolys[i_ind+1] + b*lepolys[i_ind+2])

				error = np.max(u_sol - u_old)
				# print("Error:", error)
				# print("Tolerance:", tolerance)
				u_old = u_sol
				iterations += 1
			# print(f"Number of Iterations: {iterations}")
			u = u_sol

			# cummulative_error = 0
			# for l in range(5):
			# 	diffusion = -epsilon*(4*l+6)*(-1)*alphas[l]
			# 	temp = 0.5*u**2*D@(lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2])
			# 	convection = np.sum(2*temp/(N*(N+1))/lepolys[N]**2)
			# 	rhs = np.sum(2*force*(lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2])/(N*(N+1))/(lepolys[N]**2))
			# 	cummulative_error += np.abs(diffusion - convection - rhs)
			# print(f"Cummulative Error: {cummulative_error}")
			
			# import matplotlib.pyplot as plt
			# plt.figure(figsize=(10,6))
			# plt.plot(x, u, label='$u$')
			# plt.plot(x, f_, label='$f$')
			# plt.legend(shadow=True)
			# plt.xlim(-1, 1)
			# plt.grid(alpha=0.618)
			# plt.show()
			# exit()
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
	a, b = 0, -1
	return loop(N, epsilon, size, lepolys, eps_flag, equation)


# epsilon = np.random.unform(1E0, 1E-6, SIZE)
# data = []
# for i in tqdm(range(SIZE)):
# 	# x, u = lg.lg_1d_enriched(N, epsilon[i])
# 	x, u, f, a = create(N, epsilon)
# 	data.append([x,u,f,a])

data = create_fast(N, EPSILON, SIZE, EPS_FLAG, EQUATION)
data = np.array(data)

save_obj(data, f'{SIZE}N{N}', EQUATION)
