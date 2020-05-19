#training.py
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
import network
from data_loader import LGDataset, show_solution, online_mean_and_sd
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
# transform = transforms.Compose([transforms.Normalize([0.124646], [0.9978144168])])
lg_dataset = LGDataset(pickle_file='10000')
# first, second = online_mean_and_sd(pickle_file='10000')
# print("First:", first.mean().item())
# print("Second:", second.mean().item())
# end

# N is batch size; D_in is input dimension; D_out is output dimension.
N, D_in, Filters, D_out = 64, 64, 32, 64
#Define transform on data to normalize
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model = network.Net(D_in, Filters, D_out)
model.to(device)
# Construct our loss function and an Optimizer.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)


for epoch in tqdm(range(50)):
	for batch_idx, sample_batch in enumerate(trainloader):
		x = Variable(sample_batch['f']).to(device).permute(0,2,1)
		y = Variable(sample_batch['u']).to(device).permute(0,2,1)
		optimizer.zero_grad()
		y_pred = model(x)
		loss = criterion(y_pred, y)
		loss.backward()
		optimizer.step()
	print(f"Loss: {loss}")

#Get out of sample data
# test_data = LGDataset(pickle_file='1000')
# testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=True)
# for batch_idx, sample_batch in enumerate(testloader):
# 		x, y = Variable(sample_batch['f']).to(device), Variable(sample_batch['u']).to(device)
# 		break 

def relative_l2(measured, theoretical):
	return np.linalg.norm((measured-theoretical)/np.linalg.norm(theoretical))

for _ in range(3):
	yhat = y_pred[_,:,0].to('cpu').detach().numpy()
	f = x[_,:,0].to('cpu').detach().numpy()
	u = y[_,:,0].to('cpu').detach().numpy()
	error = relative_l2(yhat, u)
	plt.figure(_, figsize=(10,6))
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.title(f'Example {_+1}\nRelative $L_2$ Error: {error}')
	plt.plot(u, 'r-', label='$u$')
	plt.plot(yhat, 'b--', label='$\\hat{u}$')
	plt.plot(f, 'g', label='$f$')
	plt.legend(shadow=True)
plt.show()
