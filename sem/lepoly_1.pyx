cimport numpy

cpdef double[:] lepoly_1(int n, double[:] x):          
	cdef int k
	if n == 0:
		return numpy.ones_like(x)
	elif n == 1:
		return x
	elif n >= 2:
		polylst = numpy.ones_like(x)
		poly = x            
		for k in range(2,n+1):
			polyn = ((2*k-1)*x*poly-(k-1)*polylst)/k
			polylst, poly = poly, polyn
		return polyn
	