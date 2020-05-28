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


def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)

def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  
N, D_in, Filters, D_out = 1000, 1, 32, 64

# #Get out of sample data
FILE = '1000'
# norm_f = normalize(pickle_file=FILE, dim='f')
# norm_f = (norm_f[0].mean().item(), norm_f[1].mean().item())
# print(f"f Mean: {norm_f[0]}\nSDev: {norm_f[1]}")
# transform_f = transforms.Compose([transforms.Normalize([norm_f[0]], [norm_f[1]])])
test_data = LGDataset(pickle_file=FILE)
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
uhat = u_pred[0,:].detach().numpy()
ff = sample_batch['f'][0,0,:].detach().numpy()
uu = sample_batch['u'][0,0,:].detach().numpy()
mae_error = mae(uhat, uu)
l2_error = relative_l2(uhat, uu)
plt.figure(figsize=(10,6))
plt.title(f'Example\nMAE Error: {np.round(mae_error, 6)}\nRel. $L_2$ Error: {np.round(l2_error, 6)}')
plt.plot(xx, uu, 'r-', label='$u$')
plt.plot(xx, uhat, 'bo', mfc='none', label='$\\hat{u}$')
plt.plot(xx, ff, 'g', label='$f$')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig('./pics/out_of_sample.png')
# plt.show()
plt.close()
