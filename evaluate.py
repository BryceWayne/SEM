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

def plotter(xx, sample, T):
	uhat = T[0,:].detach().numpy()
	ff = sample['f'][0,0,:].detach().numpy()
	uu = sample['u'][0,0,:].detach().numpy()
	mae_error = mae(uhat, uu)
	l2_error = relative_l2(uhat, uu)
	plt.figure(figsize=(10,6))
	plt.title(f'Example\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
	plt.plot(xx, uu, 'r-', label='$u$')
	plt.plot(xx, uhat, 'bo', label='$\\hat{u}$')
	plt.plot(xx, ff, 'g', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.show()

def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)

def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  
N, D_in, Filters, D_out = 100, 1, 32, 64

# #Get out of sample data
FILE = '1000'
norm_f = normalize(pickle_file=FILE, dim='f')
norm_f = (norm_f[0].mean().item(), norm_f[1].mean().item())
print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
test_data = LGDataset(pickle_file=FILE, transform_f=transform_f)
testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
for batch_idx, sample_batch in enumerate(testloader):
	f = Variable(sample_batch['f'])
	print("Got a sample.")	
	break 

# # LOAD MODEL
model = network.Net(D_in, Filters, D_out)
model.load_state_dict(torch.load('./model.pt'))
model.eval()
u_pred = model(f)
xx = sample_batch['x'][0,0,:]
plotter(xx, sample_batch, u_pred)



# L = 2
# for _ in range(L):
# 	ff = f[_,0,:].to('cpu').detach().numpy()
# 	xx = sample_batch['x'][_,0,:].numpy()
# 	uhat = u_pred[_,:].to('cpu').detach().numpy()
# 	uu = u[_,:].to('cpu').detach().numpy()
# 	mae_error = mae(uhat, uu)
# 	l2_error = relative_l2(uhat, uu)
# 	plt.figure(L + _ + 1, figsize=(10,6))
# 	plt.xlim(-1,1)
# 	plt.grid(alpha=0.618)
# 	plt.xlabel('$x$')
# 	plt.ylabel('$u$')
# 	plt.title(f'Example {_+1}\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
# 	plt.plot(xx, uu, 'r-', label='$u$')
# 	plt.plot(xx, uhat, 'b--', label='$\\hat{u}$')
# 	plt.plot(xx, ff, 'g', label='$f$')
# 	plt.legend(shadow=True)
# 	plt.show()

# #Get out of sample data
# FILE = '1000'
# norm_f = normalize(pickle_file=FILE, dim='f')
# norm_f = (norm_f[0].mean().item(), norm_f[1].mean().item())
# print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
# transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
# test_data = LGDataset(pickle_file=FILE, transform_f=transform_f)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
# for batch_idx, sample_batch in enumerate(testloader):
# 		f = Variable(sample_batch['f']).to(device)
# 		u = Variable(sample_batch['u']).to(device)
# 		break 


# model2.eval()
# optimizer2.zero_grad()
# u_pred = model2(f)
# L = 2
# for _ in range(L):
# 	ff = f[_,0,:].to('cpu').detach().numpy()
# 	xx = sample_batch['x'][_,0,:].numpy()
# 	uhat = u_pred[_,:].to('cpu').detach().numpy()
# 	uu = u[_,:].to('cpu').detach().numpy()
# 	mae_error = mae(uhat, uu)
# 	l2_error = relative_l2(uhat, uu)
# 	plt.figure(L + _ + 1, figsize=(10,6))
# 	plt.xlim(-1,1)
# 	plt.grid(alpha=0.618)
# 	plt.xlabel('$x$')
# 	plt.ylabel('$u$')
# 	plt.title(f'Example {_+1}\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
# 	plt.plot(xx, uu[0], 'r-', label='$u$')
# 	plt.plot(xx, uhat, 'b--', label='$\\hat{u}$')
# 	plt.plot(xx, ff, 'g', label='$f$')
# 	plt.legend(shadow=True)
# 	plt.show()
