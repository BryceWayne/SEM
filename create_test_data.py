import pickle
import os
import LG_1d as lg
import numpy as np

def save_obj(obj, name):
	cwd = os.getcwd()
	path = os.path.join(cwd,'data')
	if os.path.isdir(path) == False:
		os.makedirs('data')
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

N = 64
epsilon = np.linspace(1E0, 1E-6, 100000)

data = np.zeros_like((1000, N))
for i in range(1000):
	x, u = lg.lg_1d_enriched(N, np.random.choice(epsilon))
	print(x.shape)
	u = u.reshape(1,u.shape[0])
	print(u.shape)
	data[i,:] = [u]


save_obj(data, 'init')