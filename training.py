#training.py
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
import gc
from sem.sem import legslbndm, lepoly
import subprocess, os

gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--file", type=str, default='100N31')
parser.add_argument("--batch", type=int, default=100)
parser.add_argument("--epochs", type=int, default=101)
parser.add_argument("--sched", type=list, default=[25,50,75,100])
args = parser.parse_args()
FILE = args.file
SHAPE = int(args.file.split('N')[1]) + 1
BATCH = int(args.file.split('N')[0])
N, D_in, Filters, D_out = BATCH, 1, 32, SHAPE

def plotter(xx, sample, a_pred, u_pred, epoch):
	global D_out
	def relative_l2(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)
	def mae(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)
	ahat = a_pred[0,:].to('cpu').detach().numpy()
	aa = sample['a'][0,0,:].to('cpu').detach().numpy()
	uu = sample['u'][0,0,:].to('cpu').detach().numpy()
	ff = sample['f'][0,0,:].to('cpu').detach().numpy()
	x_ = legslbndm(D_out-2)
	xxx = np.linspace(-1,1, len(xx), endpoint=True)
	mae_error_a = mae(ahat, aa)
	l2_error_a = relative_l2(ahat, aa)
	plt.figure(1, figsize=(10,6))
	plt.title(f'Alphas Example Epoch {epoch}\n'\
		      f'Alphas MAE Error: {np.round(mae_error_a, 6)}\n'\
		      f'Alphas Rel. $L_2$ Error: {np.round(l2_error_a, 6)}')
	plt.plot(x_, aa, 'r-', mfc='none', label='$\\alpha$')
	plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
	plt.plot(xxx, ff, 'g-', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'./pics/alphas_epoch{epoch}.png')
	uhat = u_pred[0,:].to('cpu').detach().numpy()
	mae_error_u = mae(uhat, uu)
	l2_error_u = relative_l2(uhat, uu)
	xx = legslbndm(D_out)
	plt.figure(2, figsize=(10,6))
	plt.title(f'Reconstruction Example Epoch {epoch}\n'\
		      f'Reconstruction MAE Error: {np.round(mae_error_u, 6)}\n'\
		      f'Reconstruction Rel. $L_2$ Error: {np.round(l2_error_u, 6)}')
	plt.plot(xx, uu, 'r-', mfc='none', label='$u$')
	plt.plot(xx, uhat.T, 'bo', mfc='none', label='$\\hat{u}$')
	plt.plot(xxx, ff, 'g-', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'./pics/reconstruction_epoch{epoch}.png')
	plt.show()
	plt.close()

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
			# print(jj, i_ind)
			# print(sol.shape)
			# print(a.shape, lepolys[i_ind].shape, lepolys[i_ind+2].shape)
			sol += a[i_ind]*(lepolys[i_ind]-lepolys[i_ind+2])
		T[ii,:] = sol.T[0]
	return T

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

# Load the dataset
# norm_f = normalize(pickle_file=FILE, dim='f')
# norm_f = (0.1254, 0.9999)
# print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
# transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
# norm_a = normalize(pickle_file=FILE, dim='a')
# norm_a = (3.11411E-09, 0.032493)
# print(f"a Mean: {norm_a[0]}\nSDev: {norm_a[1]}")
# transform_a = transforms.Compose([transforms.Normalize([norm_a[0]], [norm_a[1]])])
try:
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
except:
	subprocess.call(f'python create_train_data.py --size {BATCH} --N {SHAPE - 1}', shell=True)
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model1 = network.Net(D_in, Filters, D_out, kernel_size=7, padding=3)
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
# XAVIER INITIALIZATION
model1.apply(weights_init)
# SEND TO GPU
model1.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
# optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-6, momentum=0.9)
optimizer1 = torch.optim.LBFGS(model1.parameters(), history_size=args.batch, tolerance_grad=1e-9, tolerance_change=1e-9)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.sched, gamma=0.9)

EPOCHS = args.epochs
for epoch in tqdm(range(EPOCHS)):
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		a = Variable(sample_batch['a']).to(device)
		u = Variable(sample_batch['u']).to(device)
		"""
		f -> alphas -> ?u
		"""
		def closure():
			global f, a, u
			if torch.is_grad_enabled():
				optimizer1.zero_grad()
			a_pred = model1(f)
			a = a.reshape(N, D_out-2)
			assert a_pred.shape == a.shape
			"""
			RECONSTRUCT SOLUTIONS
			"""
			u_pred = reconstruct(N, a_pred.clone(), lepolys)
			u_pred = torch.from_numpy(u_pred).to(device)
			u = u.reshape(N, D_out)
			assert u_pred.shape == u.shape
			"""
			COMPUTE LOSS
			"""
			if epoch < 100:
				loss1 = criterion2(a_pred, a)
			else:
				loss1 = criterion2(a_pred, a) + criterion2(u_pred, u)
			if loss1.requires_grad:
				loss1.backward()
			return a_pred, u_pred, loss1
		a_pred, u_pred, loss1 = closure()
		# print(f"\nLoss1: {np.round(float(loss1.to('cpu').detach()), 6)}")
		optimizer1.step(loss1.item)
	# scheduler1.step()
	print(f"\nLoss1: {np.round(float(loss1.to('cpu').detach()), 6)}")
	if epoch % 10 == 0 and epoch > 0:
		plotter(xx, sample_batch, a_pred, u_pred, epoch)


# SAVE MODEL
torch.save(model1.state_dict(), 'model.pt')
