import pickle
import os
import LG_1d as lg
import numpy as np
from tqdm import tqdm


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

SIZE = 100000
N = 63
epsilon = 1E-1
# epsilon = np.random.unform(1E0, 1E-6, SIZE)

data = []
for i in tqdm(range(SIZE)):
	# x, u = lg.lg_1d_enriched(N, epsilon[i])
	x, u, f, a = create(N, epsilon)
	data.append([x,u,f,a])

data = np.array(data)

save_obj(data, f'{SIZE}')

#END