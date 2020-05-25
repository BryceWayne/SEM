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
import gc
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--epochs", type=int, default=11)
parser.add_argument("--sched", type=list, default=[25,50,75,100])
args = parser.parse_args()


def plotter(xx, sample, T, epoch):
	uhat = T[0,:].to('cpu').detach().numpy()
	xx = sample['x'][0,0,:].to('cpu').detach().numpy()
	ff = sample['f'][0,0,:].to('cpu').detach().numpy()
	uu = sample['u'][0,0,:].to('cpu').detach().numpy()
	mae_error = mae(uhat, uu)
	l2_error = relative_l2(uhat, uu)
	plt.figure(figsize=(10,6))
	plt.title(f'Solution Example Epoch {epoch}\n'\
		      f'Solution MAE Error: {np.round(mae_error, 6)}\n'\
		      f'Solution Rel. $L_2$ Error: {np.round(l2_error, 6)}')
	# xx = range(len(uu))
	plt.plot(xx, uu, 'r-o', mfc='none', label='$u$')
	plt.plot(xx, uhat, 'b--', mfc='none', label='$\\hat{u}$')
	plt.plot(xx, ff, 'g', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'./pics/epoch{epoch}.png')
	# plt.show()
	plt.close()
	#####################################
	# uhat = np.append(uhat[0], np.zeros(64-D_out))
	# uhat = uhat[0]
	# u_sol = LG_1d.reconstruct(uhat)
	# u = sample['u'][0,0,:].to('cpu').detach().numpy()
	# # mae_error = mae(u_sol, u)
	# # l2_error = relative_l2(u_sol, u)
	# plt.figure(figsize=(10,6))
	# plt.title(f'Reconstruction Example Epoch {epoch}\n'\
	# 	      f'Reconstruction MAE Error: {np.round(mae_error, 6)}\n'\
	# 	      f'Reconstruction Rel. $L_2$ Error: {np.round(l2_error, 6)}')
	# x = np.linspace(-1, 1, len(u), endpoint=True)
	# plt.plot(x, u, 'r-', mfc='none', label='$u$')
	# x = np.linspace(-1, 1, len(u_sol), endpoint=True)
	# plt.plot(x, u_sol, 'b-x', mfc='none', label='$\\hat{u}$')
	# # plt.plot(xx, ff, 'g', label='$f$')
	# # plt.xlim(-1,1)
	# plt.grid(alpha=0.618)
	# plt.xlabel('$x$')
	# plt.ylabel('$u$')
	# plt.legend(shadow=True)
	# plt.savefig(f'./u{epoch}.png')
	# # plt.show()
	# plt.close()


def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)

def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

N, D_in, Filters, D_out = 5000, 1, 32, 64
FILE = '10000'
# Load the dataset
# norm_f = normalize(pickle_file=FILE, dim='f')
norm_f = (0.1254, 0.9999)
# print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
# norm_a = normalize(pickle_file=FILE, dim='a')
# norm_a = (3.11411E-09, 0.032493)
# print(f"a Mean: {norm_a[0]}\nSDev: {norm_a[1]}")
# transform_a = transforms.Compose([transforms.Normalize([norm_a[0]], [norm_a[1]])])
lg_dataset = LGDataset(pickle_file=FILE, subsample=D_out) #, transform_f= transform_f, transform_a=transform_a
# N is batch size; D_in is input dimension; D_out is output dimension.
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
model1 = network.Net(D_in, Filters, D_out)
model1.apply(weights_init)
model1.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
# optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-6, momentum=0.9)
optimizer1 = torch.optim.LBFGS(model1.parameters(), history_size=10, max_iter=5)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.sched, gamma=0.9)

EPOCHS = args.epochs
for epoch in tqdm(range(EPOCHS)):
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		# a = Variable(sample_batch['a']).reshape(N, D_out).to(device)
		u = Variable(sample_batch['u']).reshape(N, D_out).to(device)
		"""
		f -> ?alphas -> u
		"""
		def closure():
			if torch.is_grad_enabled():
				optimizer1.zero_grad()
			u_pred = model1(f)
			assert u_pred.shape == u.shape
			# RECONSTRUCT
			# f_pred = u_pred.clone()
			# for _ in range(len(u_pred[:,0])):
			# 	DE = LG_1d.reconstruct(u_pred[_,:].to('cpu').detach().numpy())
			# 	ux, uxx = LG_1d.derivs(DE)
			# 	f_pred[_,:] = torch.tensor((-1E-1*uxx-ux).T[0]).to(device)
			# f_out = f.clone().reshape(N, D_out)
			# f_pred.reshape(N, D_out)
			loss1 = criterion2(u_pred, u)# + criterion2(f_pred, f_out)
			if loss1.requires_grad:
				loss1.backward()
			return u_pred, loss1
        
		u_pred, loss1 = closure()
		# print(f"\nLoss1: {np.round(float(loss1.to('cpu').detach()), 6)}")
		optimizer1.step(loss1.item)
	# scheduler1.step()
	print(f"\nLoss1: {np.round(float(loss1.to('cpu').detach()), 6)}")
	# end
	if epoch % 10 == 0 and epoch > 0:
		xx = sample_batch['x'][0,0,:]
		plotter(xx, sample_batch, u_pred, epoch)

# SAVE MODEL
torch.save(model1.state_dict(), 'model.pt')
# # LOAD MODEL
# model = network.Net(D_in, Filters, D_out)
# model.load_state_dict(torch.load('model.pt'))
# model.eval()
