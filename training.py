#training.py
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
import net.network as network
from net.data_loader import LGDataset, show_solution, normalize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import LG_1d


# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

FILE = '50000'
# Load the dataset
norm = normalize(pickle_file=FILE)
transform = transforms.Compose([transforms.Normalize([norm[0].mean().item()], [norm[1].mean().item()])])
lg_dataset = LGDataset(pickle_file=FILE, transform=transform)

# N is batch size; D_in is input dimension; D_out is output dimension.
N, D_in, Filters, D_out = 100, 1, 32, 64
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model = network.Net(D_in, Filters, D_out)
model.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,125,175], gamma=0.1)

EPOCHS = 1
for epoch in tqdm(range(EPOCHS)):
	for batch_idx, sample_batch in tqdm(enumerate(trainloader)):
		x = Variable(sample_batch['f']).to(device)
		y = Variable(sample_batch['u']).to(device)
		optimizer.zero_grad()
		y_pred = model(x)
		assert y_pred.shape == y.shape
		# USE PREDICTED VALUES AS ALPHAS for LG-METHOD
		for _ in range(len(y_pred[:,0,0])):
			alphas = y_pred[_,0,:].to('cpu').detach().numpy()
			# print(alphas)
			# print(type(alphas))
			# print(alphas.shape)
			u = LG_1d.reconstruct(alphas)
			u = torch.from_numpy(u).view(1, 1, D_out).to(device)
			# print(u.shape)
			y_pred[_,0,:] = u
		loss = criterion1(y_pred, y)
		loss.backward()
		optimizer.step()
		scheduler.step()
	print(f"\nLoss: {np.round(float(loss.to('cpu').detach()), 6)}")

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
#Get out of sample data
# FILE = '1000'
# norm = normalize(pickle_file=FILE)
# transform = transforms.Compose([transforms.Normalize([norm[0].mean().item()], [norm[1].mean().item()])])
# test_data = LGDataset(pickle_file=FILE, transform=transform)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
# for batch_idx, sample_batch in enumerate(testloader):
# 		x = Variable(sample_batch['f']).to(device)
# 		y = Variable(sample_batch['u']).to(device)
# 		break 


# for _ in range(5):
# 	yhat = y_pred[_,0,:].to('cpu').detach().numpy()
# 	f = x[_,0,:].to('cpu').detach().numpy()
# 	u = y[_,0,:].to('cpu').detach().numpy()
# 	xx = sample_batch['x'][_,0,:].numpy()
# 	mae_error = mae(yhat, u)
# 	l2_error = relative_l2(yhat, u)
# 	plt.figure(_, figsize=(10,6))
# 	plt.grid(alpha=0.618)
# 	plt.xlabel('$x$')
# 	plt.ylabel('$y$')
# 	plt.title(f'Example {_+1}\nMAE Error: {mae_error}\nRel. $L_2$ Error: {l2_error}')
# 	plt.plot(xx, u, 'r-', label='$u$')
# 	plt.plot(xx, yhat, 'b--', label='$\\hat{u}$')
# 	plt.plot(xx, f, 'g', label='$f$')
# 	plt.legend(shadow=True)
# plt.show()
