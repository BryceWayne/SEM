#training.py
import random
import torch
from torch.autograd import Variable
import network
from data_loader import LGDataset, show_solution
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Load the dataset
lg_dataset = LGDataset(pickle_file='1000')
# N is batch size; D_in is input dimension; D_out is output dimension.
N, D_in, Filters, D_out = 64, 1, 1, 64
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model = network.Net(D_in, Filters, D_out)
model.to(device)
# Construct our loss function and an Optimizer.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)


for epoch in tqdm(range(200)):
	for batch_idx, sample_batch in enumerate(trainloader):
		x, y = Variable(sample_batch['f']).to(device), Variable(sample_batch['u']).to(device)
		optimizer.zero_grad()
		y_pred = model(x)
		loss = criterion(y_pred, y)
		loss.backward()
		optimizer.step()



for _ in range(3):
	yhat = y_pred[_,0,:].to('cpu').detach().numpy()
	f = x[_,0,:].to('cpu').detach().numpy()
	u = y[_,0,:].to('cpu').detach().numpy()
	plt.figure(_, figsize=(10,6))
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.title(f'Example {_+1}')
	plt.plot(yhat, label='$\\hat{u}$')
	plt.plot(u, label='$u$')
	plt.plot(f, label='$f$')
	plt.legend(shadow=True)
plt.show()