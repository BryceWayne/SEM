#evaluate.py
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import net.network as network
from net.data_loader import LGDataset, show_solution, normalize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import LG_1d
import argparse
import scipy as sp
from scipy.sparse import diags
from sem.sem import legslbndm, lepoly


parser = argparse.ArgumentParser("SEM")
parser.add_argument("--file", type=str, default='100N31')
args = parser.parse_args()
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)
SHAPE = int(args.file.split('N')[1]) + 1
BATCH = int(args.file.split('N')[0])
N, D_in, Filters, D_out = BATCH, 1, 32, SHAPE
xx = legslbndm(D_out)
def gen_lepolys(N, x):
	lepolys = {}
	for i in range(N+5):
		lepolys[i] = lepoly(i, x)
	return lepolys
lepolys = gen_lepolys(SHAPE, xx)
def reconstruct(N, alphas, lepolys):
	i, j = alphas.shape
	j += 2
	T = torch.zeros((i, j))
	T = T.detach().numpy()
	temp = alphas.clone().to('cpu').detach().numpy()
	for ii in range(i):
		a = temp[ii,:].reshape(j-2, 1)
		sol = np.zeros((j,1))
		for jj in range(1,j-2):
			i_ind = jj - 1
			sol += a[i_ind]*(lepolys[i_ind]-lepolys[i_ind+2])
		T[ii,:] = sol.T[0]
	return T
def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)

def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)



# #Get out of sample data
FILE = args.file
test_data = LGDataset(pickle_file=FILE, shape=SHAPE)
testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
for batch_idx, sample_batch in enumerate(testloader):
	f = Variable(sample_batch['f'])
	print("Got a sample.")	
	break 

# # LOAD MODEL
model = network.Net(D_in, Filters, D_out)
model.load_state_dict(torch.load('./model.pt'))
model.eval()
a_pred = model(f)
u_pred = reconstruct(SHAPE, a_pred, lepolys)
xx = legslbndm(SHAPE-2)
ahat = a_pred[0,:].detach().numpy()
ff = sample_batch['f'][0,0,:].detach().numpy()
aa = sample_batch['a'][0,0,:].detach().numpy()
mae_error_a = mae(ahat, aa)
l2_error_a = relative_l2(ahat, aa)
plt.figure(1, figsize=(10,6))
plt.title(f'Example\nMAE Error: {np.round(mae_error_a, 6)}\nRel. $L_2$ Error: {np.round(l2_error_a, 6)}')
plt.plot(xx, aa, 'r-', label='$u$')
plt.plot(xx, ahat, 'bo', mfc='none', label='$\\hat{u}$')
xx_ = np.linspace(-1,1, len(xx)+2, endpoint=True)
plt.plot(xx_, ff, 'g', label='$f$')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig('./pics/alpha_out_of_sample.png')
# plt.show()
plt.close()

uhat = u_pred[0,:]
uu = sample_batch['u'][0,0,:].detach().numpy()
mae_error_u = mae(uhat, uu)
l2_error_u = relative_l2(uhat, uu)
xx = legslbndm(SHAPE)
plt.figure(2, figsize=(10,6))
plt.title(f'Example\nMAE Error: {np.round(mae_error_u, 6)}\nRel. $L_2$ Error: {np.round(l2_error_u, 6)}')
plt.plot(xx, uu, 'r-', label='$u$')
plt.plot(xx, uhat, 'bo', mfc='none', label='$\\hat{u}$')
xx_ = np.linspace(-1,1, len(xx), endpoint=True)
plt.plot(xx_, ff, 'g', label='$f$')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig('./pics/reconstruction_out_of_sample.png')
# plt.show()
plt.close()
