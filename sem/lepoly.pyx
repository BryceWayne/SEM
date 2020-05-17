import numpy as np

def lepoly(n:int, x:float, nargout=1):
	cdef int k = 2
	if nargout == 1:
		if n == 0:
			return np.ones_like(x)
		elif n == 1:
			return x
		else:
			polylst = np.ones_like(x) #L_0(x)=1
			poly = x                  #L_1(x)=x
			for k in range(2,n+1):
				polyn = ((2*k-1)*x*poly-(k-1)*polylst)/k
				polylst, poly = poly, polyn
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
	