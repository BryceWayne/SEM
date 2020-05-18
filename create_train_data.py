import pickle
import os
import LG_1d as lg
import numpy as np
from tqdm import tqdm

"""
METRICS

N = 63, 1000 solutions with random forcing: 200.3 seconds
N = 63, 1000 solutions with random forcing: 1025.2 seconds
"""

def save_obj(obj, name):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data')
	if os.path.isdir(path) == False:
		os.makedirs('data')
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def create(N:int, epsilon:float):
	x, u, f = lg.lg_1d_standard(N, epsilon)
	x, u, f = x.reshape(1,x.shape[0]), u.reshape(1,u.shape[0]), f.reshape(1,f.shape[0])
	return x, u, f
SIZE = 10000
N = 63
epsilon = 1E-1
# epsilon = np.random.unform(0, 1E-6, SIZE)

data = []
for i in tqdm(range(SIZE)):
	# x, u = lg.lg_1d_enriched(N, epsilon[i])
	x, u, f = create(N, epsilon)
	data.append([x,u,f])

data = np.array(data)

save_obj(data, f'{SIZE}')

#END