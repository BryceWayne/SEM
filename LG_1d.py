from sem import sem
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cProfile

def exact(x: float, epsilon: float) -> float:
	return 2*(np.exp(-(x+1)/epsilon)-1)/(np.exp(-2/epsilon)-1)-(x+1)
def plotter(x, y, diff=False):
	exact_sol = exact(x, epsilon).T[0]
	error = np.round(np.linalg.norm(y-exact_sol)/np.linalg.norm(exact_sol), 16)
	plt.figure(1, figsize=(10,6))
	plt.plot(x, exact_sol, 'b', label='Exact')
	plt.plot(x, y, 'ro--', markersize=3, label='Approx')
	plt.grid(alpha=True)
	plt.legend(shadow=True)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.title(f'Parameters: N={N}, $\\varepsilon$={epsilon}\nRelative $L_2$ Error={error}')
	plt.show()
	if diff == True:
		plt.figure(2, figsize=(10,6))
		plt.plot(x, exact_sol-y, 'bo-', label='Diff')
		plt.grid(alpha=True)
		plt.legend(shadow=True)
		plt.xlabel('$x$')
		plt.ylabel('DIFF')
		plt.show()
def lg_1d_standard(N:int, epsilon:float) -> float:
	x = sem.legslbndm(N+1)
	D = sem.legslbdiff(N+1, x)
	a = 0
	b = -1
	def func(t: float) -> float:
		return np.ones_like(t)

	f = func(x)
	s_diag = np.zeros((N-1,1))
	M = np.zeros((N-1,N-1))
	for ii in range(1, N):
		k = ii - 1
		s_diag[ii-1] = -(4*k+6)*b
		phi_k_M = D@(sem.lepoly(k,x) + a*sem.lepoly(k+1,x) + b*sem.lepoly(k+2,x))
		for jj in range(1,N):
			if abs(ii-jj) <=2:
				l = jj-1
				psi_l_M = sem.lepoly(l,x) + a*sem.lepoly(l+1,x) + b*sem.lepoly(l+2,x)
				M[jj-1,ii-1] = sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(sem.lepoly(N,x)**2))

	S = s_diag*np.eye(N-1)
	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i-1
		g[i-1] = (2*k+1)/(N*(N+1))*sum(f*(sem.lepoly(k,x))/(sem.lepoly(N,x)**2))
	g[N-1] = 1/(N+1)*sum(f/sem.lepoly(N,x))

	bar_f = np.zeros((N-1,))
	for i in range(1,N):
		k = i-1
		bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

	Mass = epsilon*S-M
	u = np.linalg.solve(Mass, bar_f)
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
			L = sem.lepoly(k,x)
			_ += g[j-1]*L[i-1]
		_ = _[0]
		u[i-1] = _

	return x, u
def lg_1d_enriched(N:int, epsilon:float) -> float:
	sigma = 1
	x = sem.legslbndm(N+1)
	D = sem.legslbdiff(N+1, x)
	a = 0
	b = -1
	def func(t: float) -> float:
		return np.ones_like(t)

	f = 0.5*func(x)

	phi = get_phi(N, x, sigma, epsilon)
	residual = -(np.exp(-(sigma+1)/epsilon)-1)/(sigma+1)
	
	s_diag = np.zeros((N-1,1))
	M = np.zeros((N-1,N-1))
	for ii in range(1, N):
		k = ii - 1
		s_diag[ii-1] = -(4*k+6)*b
		phi_k_M = D@(sem.lepoly(k,x) + a*sem.lepoly(k+1,x) + b*sem.lepoly(k+2,x))
		for jj in range(1,N):
			if abs(ii-jj) <=2:
				l = jj-1
				psi_l_M = sem.lepoly(l,x) + a*sem.lepoly(l+1,x) + b*sem.lepoly(l+2,x)
				M[jj-1,ii-1] = sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(sem.lepoly(N,x)**2))

	S = s_diag*np.eye(N-1)
	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i-1
		g[i-1] = (2*k+1)/(N*(N+1))*sum(f*(sem.lepoly(k,x))/(sem.lepoly(N,x)**2))
	g[N-1] = 1/(N+1)*sum(f/sem.lepoly(N,x))

	bar_f = np.zeros((N-1,))
	for i in range(1,N):
		k = i-1
		bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

	Mass = epsilon*S-M
	u = np.linalg.solve(Mass, bar_f)
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
			L = sem.lepoly(k,x)
			_ += g[j-1]*L[i-1]
		_ = _[0]
		u[i-1] = _

	return x, u


N, epsilon = 64, 1E-1
# cProfil.run('lg_1d_standard(N, epsilon)')
x, sol = lg_1d_standard(N, epsilon)
# x, sol = lg_1d_enriched(N, epsilon)
plotter(x, sol)
