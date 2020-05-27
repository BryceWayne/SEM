import numpy as np
import scipy as sp
from scipy.sparse import diags
from pprint import pprint
import pickle, os
# from Cython.Build import cythonize

def load_obj(pickle_file):
	with open('data/' + pickle_file + '.pkl', 'rb') as f:
		return pickle.load(f)
def save_obj(obj, name):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data')
	if os.path.isdir(path) == False:
		os.makedirs('data')
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def lepoly(n:int, x:np.ndarray, nargout=1) -> np.ndarray:
	if nargout == 1:
		if n == 0:
			return np.ones_like(x)
		elif n == 1:
			return x
		else:
			# try:
			# 	polyn = load_obj(f"LG{n}")
			# 	return polyn
			# except:
			polylst = np.ones_like(x) #L_0(x)=1
			poly = x                  #L_1(x)=x
			for k in range(2,n+1):
				polyn = ((2*k-1)*x*poly-(k-1)*polylst)/k
				polylst, poly = poly, polyn
			# save_obj(polyn, name=f'LG{n}')
			return polyn
	elif nargout == 2:
		if n == 0:
			return np.zeros_like(x), np.ones_like(x)
		elif n == 1:
			return np.ones_like(x), x
		else:
			polylst, pderlst = np.ones_like(x), np.zeros_like(x)
			poly, pder = x, np.ones_like(x)
			for k in range(2,n+1):
				polyn=((2*k-1)*x*poly-(k-1)*polylst)/k # kL_k(x)=(2k-1)xL_{k-1}(x)-(k-1)L_{k-2}(x)
				pdern=pderlst+(2*k-1)*poly             # L_k'(x)=L_{k-2}'(x)+(2k-1)L_{k-1}(x)
				polylst, poly = poly, polyn
				pderlst, pder = pder, pdern
			return pdern, polyn
def legslbndm(n:int) -> np.ndarray:
	if n <= 1:
		print("n should be bigger than 1")
		return np.array([[]])
	elif n == 2:
		return np.array([[-1, 1]]).T
	elif n == 3:
		return np.array([[-1, 0, 1]]).T
	else:
		av = np.zeros((1,n-2)).T
		j = np.array([list(range(1, n-2))])
		bv = j*(j+2)/((2*j+1)*(2*j+3))
		A = diags([np.sqrt(bv), 0, np.sqrt(bv)], [-1, 0, 1], shape=(n-2,n-2))
		z = np.sort(np.linalg.eig(A.toarray())[0])
		z = [-1, *z, 1]
		z = np.array(z).T
		return z.reshape(n, 1)
def legslbdiff(n:int, x:np.ndarray) -> np.ndarray:
	if n == 0:
		return np.array([[]])
	xx = x
	y = lepoly(n-1, xx)
	nx = len(x)
	if x.shape[1] > x.shape[0]:
		y = y.T
		xx = x.T
	D = (xx/y)@y.T - (1/y)@(xx*y).T
	D += np.eye(n)
	D = 1/D
	D = D - np.eye(n)
	D[0,0] = -n*(n-1)/4
	D[n-1,n-1] = -D[0,0]
	return D
def get_phi(N:int, x:np.ndarray, sigma:float, epsilon:float) -> np.ndarray:
	sol = np.zeros_like(x)
	for i in range(1,N+2):
		if x[i-1] < sigma:
			sol[i-1] = 1- np.exp(-(1+x[i-1])/epsilon)-(1-np.exp(-(1+sigma)/epsilon))*(x[i-1]+1)/(1+sigma)
		else:
			sol[i-1] = 0
	return sol
