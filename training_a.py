#training.py
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import LG_1d
import argparse
import scipy as sp
from scipy.sparse import diags
import gc
import subprocess, os
import net.network as network
from net.data_loader import *
from sem.sem import *
from plotting import *
from reconstruct import *
import datetime
import pandas as pd
import time


gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--file", type=str, default='500N31')
parser.add_argument("--batch", type=int, default=500)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--ks", type=int, default=3)
parser.add_argument("--data", type=bool, default=True)
args = parser.parse_args()


KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
FILE = args.file
BATCH = int(args.file.split('N')[0])
SHAPE = int(args.file.split('N')[1]) + 1
N, D_in, Filters, D_out = BATCH, 1, 32, SHAPE
EPOCHS = args.epochs
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
PATH = os.path.join(FILE, cur_time)

try:
	os.mkdir(FILE)
	exists = False
except:
	exists = True
os.mkdir(PATH)
os.mkdir(os.path.join(PATH,'pics'))

xx = legslbndm(SHAPE)
lepolys = gen_lepolys(SHAPE, xx)
lepoly_x = dx(SHAPE, xx, lepolys)
lepoly_xx = dxx(SHAPE, xx, lepolys)
phi = basis(SHAPE, lepolys)
phi_x = basis_x(SHAPE, phi, lepoly_x)
phi_xx = basis_xx(SHAPE, phi_x, lepoly_xx)

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)


# Load the dataset
try:
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
except:
	subprocess.call(f'python create_train_data.py --size {BATCH} --N {SHAPE - 1}', shell=True)
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model1 = network.NetA(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING)

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model1.apply(weights_init)
# SEND TO GPU
model1.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
optimizer1 = torch.optim.LBFGS(model1.parameters(), history_size=10, tolerance_grad=1e-16, tolerance_change=1e-16, max_eval=10)
# optimizer1 = torch.optim.SGD(model1.parameters(), lr=1E-4)

BEST_LOSS = 9E32
losses = []
time0 = time.time()
for epoch in tqdm(range(1, EPOCHS+1)):
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		a = Variable(sample_batch['a']).to(device)
		u = Variable(sample_batch['u']).to(device)
		"""
		f -> alphas >> ?u
		"""
		def closure(f, a, u):
			if torch.is_grad_enabled():
				optimizer1.zero_grad()
			a_pred = model1(f)
			a = a.reshape(N, D_out-2)
			assert a_pred.shape == a.shape
			"""
			RECONSTRUCT SOLUTIONS
			"""
			# u_pred = reconstruct(a_pred, phi)
			u = u.reshape(N, D_out)
			# assert u_pred.shape == u.shape
			u_pred = None
			"""
			RECONSTRUCT ODE
			"""
			# DE = ODE2(1E-1, u_pred, a_pred, phi_x, phi_xx)
			f = f.reshape(N, D_out)
			# assert DE.shape == f.shape
			DE = None
			"""
			WEAK FORM
			"""
			# LHS, RHS = weak_form1(1E-1, SHAPE, f, u_pred, a_pred, lepolys, phi_x)
			# LHS, RHS = weak_form2(1E-1, SHAPE, f, u, a_pred, lepolys, phi, phi_x)
			"""
			COMPUTE LOSS
			"""
			loss = criterion2(a_pred, a)# + criterion1(u_pred, u) + criterion1(DE, f) + criterion1(LHS, RHS) # + criterion1(DE, f)		
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, DE, loss
		a_pred, u_pred, DE, loss = closure(f, a, u)
		optimizer1.step(loss.item)
		current_loss = np.round(float(loss.to('cpu').detach()), 8)
		losses.append(current_loss) 
	print(f"\tLoss: {current_loss}")
	if epoch % int(.1*EPOCHS) == 0:
		u_pred = reconstruct(a_pred, phi)
		DE = ODE2(1E-1, u_pred, a_pred, phi_x, phi_xx)
		plotter(xx, sample_batch, epoch, a=a_pred, u=u_pred, DE=DE, title='a', ks=KERNEL_SIZE, path=PATH)
	if current_loss < BEST_LOSS:
		torch.save(model1.state_dict(), PATH + '/model.pt')
		BEST_LOSS = current_loss


time1 = time.time()
dt = time1 - time0
avg_iter_time = np.round(dt/EPOCHS, 6)
if args.data == True:
	subprocess.call(f'python evaluate_a.py --ks {KERNEL_SIZE} --input {FILE} --path {PATH} --data True', shell=True)
else:
	subprocess.call(f'python evaluate_a.py --ks {KERNEL_SIZE} --input {FILE} --path {PATH}', shell=True)
loss_plot(losses, FILE, EPOCHS, SHAPE, KERNEL_SIZE, BEST_LOSS, title='a', path=PATH)
gc.collect()
torch.cuda.empty_cache()
if args.data == True:
	COLS = ['TIMESTAMP', 'DATASET', 'FOLDER', 'SHAPE', 'K.SIZE', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	df = pd.read_excel('temp.xlsx')
	df.at[df.index[-1],'AVG IT/S'] = float(avg_iter_time)
	df.at[df.index[-1],'LOSS'] = float(min(losses))
	df.at[df.index[-1],'EPOCHS'] = int(EPOCHS)
	df.at[df.index[-1],'BATCH'] = N
	df = df[COLS]
	_ = ['AVG IT/S', 'LOSS']
	for obj in _:
		df[obj] = df[obj].astype(float)
	df.to_excel('temp.xlsx')
