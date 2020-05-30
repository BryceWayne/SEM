from sem.sem import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cProfile


def exact(x:np.ndarray, epsilon:float, enriched=False) -> np.ndarray:
	if enriched == False:
		return 2*(np.exp(-(x+1)/epsilon)-1)/(np.exp(-2/epsilon)-1)-(x+1)
	else:
		return (np.exp(-(x+1)/epsilon)-1)/(np.exp(-2/epsilon)-1)-(x+1)/2
def plotter(x:np.ndarray, y:np.ndarray, exact_flag = False, enriched=False, diff=False):
	if enriched == False:
		exact_sol = exact(x, epsilon).T[0]
	else:
		exact_sol = exact(x, epsilon, enriched=True).T[0]

	if exact_flag == True:
		error = np.round(np.linalg.norm(y-exact_sol)/np.linalg.norm(exact_sol), 18)
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

	if exact_flag == False:
		plt.figure(1, figsize=(10,6))
		plt.plot(x, y, 'ro--', markersize=3, label='Approx')
		plt.grid(alpha=True)
		plt.legend(shadow=True)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.title(f'Parameters: N={N}, $\\varepsilon$={epsilon}')
		plt.show()
def lg_1d_standard(N:int, epsilon:float, exact_flag = False) -> np.ndarray:
	x = legslbndm(N+1)
	D = legslbdiff(N+1, x)
	a = 0
	b = -1
	def func(t: float, exact_flag) -> float:
		# Random force
		if exact_flag == False:
			m = 2*np.random.rand(4)
			return m[0]*np.sin(m[1]*np.pi*t) + m[2]*np.cos(m[3]*np.pi*t)
		# Exact solution
		if exact_flag == True:
			return np.ones_like(t)

	f = func(x, exact_flag)
	s_diag = np.zeros((N-1,1))
	M = np.zeros((N-1,N-1))
	for ii in range(1, N):
		k = ii - 1
		s_diag[ii-1] = -(4*k+6)*b
		phi_k_M = D@(lepoly(k,x) + a*lepoly(k+1,x) + b*lepoly(k+2,x))
		for jj in range(1,N):
			if abs(ii-jj) <=2:
				l = jj-1
				psi_l_M = lepoly(l,x) + a*lepoly(l+1,x) + b*lepoly(l+2,x)
				M[jj-1,ii-1] = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepoly(N,x)**2))

	S = s_diag*np.eye(N-1)
	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i-1
		g[i-1] = (2*k+1)/(N*(N+1))*np.sum(f*(lepoly(k,x))/(lepoly(N,x)**2))
	g[N-1] = 1/(N+1)*np.sum(f/lepoly(N,x))

	bar_f = np.zeros((N-1,))
	for i in range(1,N):
		k = i-1
		bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

	Mass = epsilon*S-M
	u = np.linalg.solve(Mass, bar_f)
	alphas = np.copy(g)
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
			L = lepoly(k,x)
			_ += g[j-1]*L[i-1]
		_ = _[0]
		u[i-1] = _
	return x, u, f, alphas
def lg_1d_enriched(N:int, epsilon:float, exact_flag = False) -> np.ndarray:
	sigma = 1
	x = legslbndm(N+1)
	D = legslbdiff(N+1, x)
	a = 0
	b = -1
	def func(t: float, exact_flag: bool) -> float:
		# Random force
		if exact_flag == False:
			m = 2*np.random.rand(4)
			return m[0]*np.sin(m[1]*np.pi*t) + m[2]*np.cos(m[3]*np.pi*t)
		# Exact solution
		if exact_flag == True:
			return 0.5*np.ones_like(t)

	f = func(x, exact_flag)

	phi = get_phi(N, x, sigma, epsilon)
	residual = -(np.exp(-(sigma+1)/epsilon)-1)/(sigma+1)
	
	S, M = np.zeros((N-1,N-1)), np.zeros((N-1,N-1))
	a_12, a_21 = np.zeros((N-1,)), np.zeros((N-1,))
	for ii in range(1, N):
		k = ii - 1

		phi_k_a12 = lepoly(k,x) + a*lepoly(k+1,x) + b*lepoly(k+2,x)
		phi_k_M = D@phi_k_a12
		phi_k_s = D@phi_k_M
		for jj in range(1,N):
			if abs(ii-jj) <=2:
				l = jj-1
				psi_l_M = lepoly(l,x) + a*lepoly(l+1,x) + b*lepoly(l+2,x)
				element = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepoly(N,x)**2))
				M[jj-1,ii-1] = np.sum((psi_l_M*phi_k_M)*2/(N*(N+1))/(lepoly(N,x)**2))

			if ii<=jj-2:
			    q = jj-1
			    phi_q = lepoly(q,x) + a*lepoly(q+1,x) + b*lepoly(q+2,x)
			    S[jj-1,ii-1] = np.sum(phi_k_s*phi_q*2/(N*(N+1))/(lepoly(N,x)**2))

		a_12[ii-1] = np.sum(residual*phi_k_a12*2/(N*(N+1))/(lepoly(N,x)**2))
		a_21[ii-1] = np.sum((-epsilon*phi_k_s-phi_k_M)*phi*2/(N*(N+1))/(lepoly(N,x)**2))
	a_22 = np.sum(residual*phi*2/(N*(N+1))/(lepoly(N,x)**2))

	Mass = epsilon*S-M
	a_12 = a_12.reshape(a_12.shape[0], 1)
	Mass = np.hstack([Mass, a_12])
	a_21 = np.append(a_21, 0)
	a_21 = a_21.reshape(1,a_21.shape[0])
	Mass = np.vstack([Mass, a_21])
	Mass[N-1,N-1] = a_22

	g = np.zeros((N+1,))
	for i in range(1,N+1):
		k = i-1
		g[i-1] = (2*k+1)/(N*(N+1))*np.sum(f*(lepoly(k,x))/lepoly(N,x)**2)

	g[N] = 1/(N+1)*np.sum(f/lepoly(N,x))

	bar_f = np.zeros((N,))
	for i in range(1, N):
		k = i-1
		bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

	bar_f_end = 0
	for ii in range(1,N+2):
		k = ii-1
		bar_f_end += g[ii-1]*2/(N*(N+1))*np.sum(phi*(lepoly(k,x))/lepoly(N,x)**2)

	bar_f[N-1] = bar_f_end

	u_temp = np.linalg.solve(Mass,bar_f)
	u_temp = u_temp.reshape(u_temp.shape[0],1)
	u_sol = np.zeros((N+1,))
	for ij in range(1,N):
		i_ind = ij - 1
		element = u_temp[ij-1,0]*(lepoly(i_ind,x) + a*lepoly(i_ind+1,x) + b*lepoly(i_ind+2,x))
		u_sol += element.T[0]
	u = u_sol + (u_temp[N-1]*phi).T[0]
	return x, u
def DE(t: float, x: np.ndarray) -> np.ndarray:
    return x*(1-x)
def RK(t: float, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = DE(t, x)*dt
    k2 = DE(t, x+0.5*k1)*dt
    k3 = DE(t, x+0.5*k2)*dt
    k4 = DE(t, x+k3)*dt
    return x+1/6*(k1+2*k2+2*k3+k4)
def rk4(x: np.ndarray, t0=0, dt=0.1) -> np.ndarray:
	pass
def reconstruct(alphas:np.ndarray):
	a, b, N = 0, -1, len(alphas) - 1
	x = legslbndm(N+1)
	u_sol = np.zeros((N+1,))
	for ij in range(1,N):
		i_ind = ij - 1
		# print(i_ind)
		# print(len(x))
		element = alphas[ij-1]*(lepoly(i_ind,x) + a*lepoly(i_ind+1,x) + b*lepoly(i_ind+2,x))
		u_sol += element.T[0]
	return u_sol
def derivs(u:np.ndarray, N=63):
	N = len(u) - 1
	x = legslbndm(N+1)
	D = legslbdiff(N+1, x)
	u = u.reshape(N+1,1)
	ux = D@u
	uxx = D@ux
	return ux, uxx

def debug():
	N, epsilon = 32, 1E-3
	profile = False
	enriched = True
	plot = True
	exact_flag = False

	if enriched == False:
		if profile == True:
			cProfile.run('lg_1d_standard(N, epsilon)')
		x, sol, f = lg_1d_standard(N, epsilon, exact_flag)
		if plot == True:
			plotter(x, sol, exact_flag)
	elif enriched == True:
		if profile == True:
			cProfile.run('lg_1d_enriched(N, epsilon)')
		x, sol = lg_1d_enriched(N, epsilon, exact_flag)
		if plot == True:
			plotter(x, sol, exact_flag, enriched=True)

