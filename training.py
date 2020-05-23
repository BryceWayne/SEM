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


def plotter(xx, sample, T):
	ahat = T[0,:].to('cpu').detach().numpy()
	f = sample['f'][0,0,:].to('cpu').detach().numpy()
	a = sample['a'][0,0,:].to('cpu').detach().numpy()
	plt.figure(figsize=(10,6))
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.title(f'Example')
	plt.plot(xx, a, 'r-', label='$\\alpha$')
	plt.plot(xx, ahat, 'b--', label='$\\hat{\\alpha}$')
	# plt.plot(xx, f, 'g', label='$f$')
	plt.legend(shadow=True)
	plt.show()
# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

FILE = '10000'
# Load the dataset
# norm_f = normalize(pickle_file=FILE, dim='f')
norm_f = (0.1254, 0.9999)
print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
# norm_a = normalize(pickle_file=FILE, dim='a')
norm_a = (3.1141098588705063E-09, 0.032493)
print(f"a Mean: {norm_a[0]}\nSDev: {norm_a[1]}")
transform_a = transforms.Compose([transforms.Normalize([norm_a[0]], [norm_a[1]])])
lg_dataset = LGDataset(pickle_file=FILE, transform_f= transform_f, transform_a=transform_a) #, transform=transform
# N is batch size; D_in is input dimension; D_out is output dimension.
N, D_in, Filters, D_out = 100, 1, 32, 64
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
model1 = network.Net(D_in, Filters, D_out)
model2 = network.U(D_in, Filters, D_out)
model1.apply(weights_init)
model2.apply(weights_init)
model1.to(device)
model2.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-5, momentum=0.9)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-5, momentum=0.9)
# optimizer2a = torch.optim.LBFGS(model2.parameters(), history_size=N, max_iter=4)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[10,20,30,40,45], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[20,40,60,80], gamma=0.1)

EPOCHS = 100
for epoch in tqdm(range(EPOCHS)):
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		a = Variable(sample_batch['a']).reshape(N, D_out).to(device)
		u = Variable(sample_batch['u']).reshape(N, D_out).to(device)
		"""
		f -> alphas -> u
		"""
		# PREDICT ALPHAS
		# optimizer1.zero_grad()
		# a_pred = model1(f)
		# assert a_pred.shape == a.shape
		# loss1 = criterion2(a_pred, a)
		# RECONSTRUCT U
		# a_pred = a_pred.reshape(N, 1, D_out)
		optimizer2.zero_grad()
		u_pred = model2(f)
		assert u_pred.shape == u.shape
		# 1000 EPOCHS w just MSE on u ######################
		#
		####################################################
		loss2 = criterion1(u_pred, u) + criterion2(u_pred, u)
		# loss1.backward(retain_graph=True)
		loss2.backward()
		# optimizer1.step()
		optimizer2.step()
	# scheduler1.step()
	scheduler2.step()
	# print(f"\nLoss1: {np.round(float(loss1.to('cpu').detach()), 6)}")
	print(f"Loss: {np.round(float(loss2.to('cpu').detach()), 6)}")
	# if epoch % 10 == 0 and epoch > 0:
	# 	xx = sample_batch['x'][0,0,:]
	# 	plotter(xx, sample_batch, a_pred)

# SAVE MODEL
# torch.save(model.state_dict(), 'model.pt')
# # LOAD MODEL
# model = network.Net(D_in, Filters, D_out)
# model.load_state_dict(torch.load('model.pt'))
# model.eval()

def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)

def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)


L = 2
for _ in range(L):
	ff = f[_,0,:].to('cpu').detach().numpy()
	xx = sample_batch['x'][_,0,:].numpy()
	uhat = u_pred[_,:].to('cpu').detach().numpy()
	uu = u[_,:].to('cpu').detach().numpy()
	mae_error = mae(uhat, uu)
	l2_error = relative_l2(uhat, uu)
	plt.figure(L + _ + 1, figsize=(10,6))
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$u$')
	plt.title(f'Example {_+1}\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
	plt.plot(xx, uu, 'r-', label='$u$')
	plt.plot(xx, uhat, 'b--', label='$\\hat{u}$')
	plt.plot(xx, ff, 'g', label='$f$')
	plt.legend(shadow=True)
	plt.show()

#Get out of sample data
FILE = '1000'
norm_f = normalize(pickle_file=FILE, dim='f')
norm_f = (norm_f[0].mean().item(), norm_f[1].mean().item())
print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
test_data = LGDataset(pickle_file=FILE, transform_f=transform)
testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
for batch_idx, sample_batch in enumerate(testloader):
		f = Variable(sample_batch['f']).to(device)
		u = Variable(sample_batch['u']).to(device)
		break 


model2.eval()
optimizer2.zero_grad()
u_pred = model2(f)
L = 2
for _ in range(L):
	ff = f[_,0,:].to('cpu').detach().numpy()
	xx = sample_batch['x'][_,0,:].numpy()
	uhat = u_pred[_,:].to('cpu').detach().numpy()
	uu = u[_,:].to('cpu').detach().numpy()
	mae_error = mae(uhat, uu)
	l2_error = relative_l2(uhat, uu)
	plt.figure(L + _ + 1, figsize=(10,6))
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$u$')
	plt.title(f'Example {_+1}\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
	plt.plot(xx, uu, 'r-', label='$u$')
	plt.plot(xx, uhat, 'b--', label='$\\hat{u}$')
	plt.plot(xx, ff, 'g', label='$f$')
	plt.legend(shadow=True)
	plt.show()
