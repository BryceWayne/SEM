import pickle
import os
import LG_1d as lg
from sem import sem as sem
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--size", type=int, default=10000)
parser.add_argument("--N", type=int, default=31)
parser.add_argument("--eps", type=bool, default=False)
args = parser.parse_args()


def save_obj(obj, name):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data')
	if os.path.isdir(path) == False:
		os.makedirs('data')
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def create(N:int, epsilon:float):
	x, u, f, a = lg.lg_1d_standard(N, epsilon)
	x = x.reshape(1,x.shape[0])
	u = u.reshape(1,u.shape[0])
	f = f.reshape(1,f.shape[0])
	a = a.reshape(1,a.shape[0])
	return x, u, f, a


def create_fast(N:int, epsilon:float, size:int, eps_flag=False):
	def func(t: float) -> float:
		# Random force
		m = 2*np.random.rand(4) - 1
		f = m[0]*np.sin(m[1]*np.pi*t) + m[2]*np.cos(m[3]*np.pi*t)
		return f, m

	def gen_lepolys(N, x):
		lepolys = {}
		for i in range(N+3):
			lepolys[i] = sem.lepoly(i, x)
		return lepolys

	def generate(x, D, a, b, lepolys, epsilon):
		f, params = func(x)
		s_diag = np.zeros((N-1,1))
		M = np.zeros((N-1,N-1))
		for ii in range(1, N):
			k = ii - 1
			s_diag[ii-1] = -(4*k+6)*b
			phi_k_M = D@(lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2])
			for jj in range(1,N):
				if abs(ii-jj) <=2:
					l = jj-1
					psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
					M[jj-1,ii-1] = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepolys[N]**2))

		S = s_diag*np.eye(N-1)
		g = np.zeros((N+1,))
		for i in range(1,N+1):
			k = i-1
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
			_ = _[0]
			u[i-1] = _
		return u, f, alphas, params

	def loop(N, epsilon, size, lepolys, eps_flag):
		if eps_flag == True:
			epsilons = np.random.uniform(1E0, 1E-6, SIZE)
		data = []
		U, F, ALPHAS, PARAMS = [], [], [], []
		for n in tqdm(range(size)):
			if eps_flag == True:
				epsilon = epsilons[n]
			u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon)
			data.append([u, f, alphas, params, epsilon])
		return data


	x = sem.legslbndm(N+1)
	D = sem.legslbdiff(N+1, x)
	a, b = 0, -1
	lepolys = gen_lepolys(N, x)
	return loop(N, epsilon, size, lepolys, eps_flag)


SIZE = args.size
N = args.N
epsilon = 1E-1
EPS_FLAG = args.eps
# epsilon = np.random.unform(1E0, 1E-6, SIZE)

# data = []
# for i in tqdm(range(SIZE)):
# 	# x, u = lg.lg_1d_enriched(N, epsilon[i])
# 	x, u, f, a = create(N, epsilon)
# 	data.append([x,u,f,a])

data = create_fast(N, epsilon, SIZE, EPS_FLAG)

data = np.array(data)

save_obj(data, f'{SIZE}N{N}')
