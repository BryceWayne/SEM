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
import subprocess, os, gc
import net.network as network
from net.data_loader import *
from net.network import *
from sem.sem import *
from plotting import *
from reconstruct import *
from data_logging import *
from evaluate_a import *
import pandas as pd
import time, datetime


# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--model", type=object, default=ResNet) #ResNet or NetA
parser.add_argument("--file", type=str, default='2000N31')
parser.add_argument("--batch", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ks", type=int, default=3)
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--data", type=bool, default=True)
args = parser.parse_args()

# VARIABLES
MODEL = args.model
DATA = args.data
KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
FILE = args.file
BATCH = int(args.file.split('N')[0])
SHAPE = int(args.file.split('N')[1]) + 1
FILTERS = args.filters
N, D_in, Filters, D_out = BATCH, 1, FILTERS, SHAPE
EPOCHS = args.epochs
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
PATH = os.path.join(FILE, cur_time)
BLOCKS = args.blocks
EPSILON = 1E-1


#CREATE PATHING
try:
	os.mkdir(FILE)
	exists = False
except:
	exists = True
os.mkdir(PATH)
os.mkdir(os.path.join(PATH,'pics'))


#CREATE BASIS VECTORS
xx = legslbndm(SHAPE)
lepolys = gen_lepolys(SHAPE, xx)
lepoly_x = dx(SHAPE, xx, lepolys)
lepoly_xx = dxx(SHAPE, xx, lepolys)
phi = basis(SHAPE, lepolys)
phi_x = basis_x(SHAPE, phi, lepoly_x)
phi_xx = basis_xx(SHAPE, phi_x, lepoly_xx)


# Load the dataset
try:
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
except:
	subprocess.call(f'python create_train_data.py --size {BATCH} --N {SHAPE - 1}', shell=True)
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)

#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model = MODEL(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)


# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model.apply(weights_init)

# Check if CUDA is available and then use it.
device = get_device()
# SEND TO GPU (or CPU)
model.to(device)

# Construct our loss function and an Optimizer.
# criterion_a = torch.nn.L1Loss()
criterion_a = torch.nn.MSELoss(reduction="sum")
# criterion_u = torch.nn.L1Loss()
criterion_u = torch.nn.MSELoss(reduction="sum")
# criterion_wf = torch.nn.L1Loss()
criterion_wf = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, tolerance_grad=1e-14, tolerance_change=1e-14, max_eval=10)
# optimizer = torch.optim.SGD(model.parameters(), lr=1E-8)


BEST_LOSS, losses = float('inf'), {'loss_a':[], 'loss_u':[], 'loss_wf':[], 'loss_train':[], 'loss_validate':[]}
time0 = time.time()
for epoch in tqdm(range(1, EPOCHS+1)):
	loss_a, loss_u, loss_wf, loss_train = 0, 0, 0, 0
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		a = Variable(sample_batch['a']).to(device)
		u = Variable(sample_batch['u']).to(device)
		def closure(f, a, u):
			if torch.is_grad_enabled():
				optimizer.zero_grad()
			a_pred = model(f)
			assert a_pred.shape == a.shape
			u_pred = reconstruct(a_pred, phi)
			assert u_pred.shape == u.shape
			# LHS, RHS = weak_form1(EPSILON, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x)
			LHS, RHS = weak_form2(EPSILON, SHAPE, f, u, a_pred, lepolys, phi, phi_x)
			loss_a = criterion_a(a_pred, a)
			loss_u = criterion_u(u_pred, u)
			loss_wf = criterion_wf(LHS, RHS)
			loss = loss_a + loss_u + loss_wf	# + criterion1(DE, f)	
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, loss, loss_a, loss_u, loss_wf
		a_pred, u_pred, loss, loss_a, loss_u, loss_wf = closure(f, a, u)
		optimizer.step(loss.item)
		loss_a += np.round(float(loss_a.to('cpu').detach()), 8)
		loss_u += np.round(float(loss_u.to('cpu').detach()), 8)
		loss_wf += np.round(float(loss_wf.to('cpu').detach()), 8)
		loss_train += np.round(float(loss.to('cpu').detach()), 8)
	loss_validate = validate(model, optimizer, EPSILON, SHAPE, FILTERS, criterion_a, criterion_u, criterion_wf, lepolys, phi, phi_x)
	losses['loss_a'].append(loss_a)
	losses['loss_u'].append(loss_u)
	losses['loss_wf'].append(loss_wf)
	losses['loss_train'].append(loss_train)
	losses['loss_validate'].append(loss_validate)
	if EPOCHS >= 10 and epoch % int(.1*EPOCHS) == 0:
		print(f"\tLoss: {loss_train}")
		if u_pred == None:
			u_pred = reconstruct(a_pred, phi)
		DE = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx)
		plotter(xx, sample_batch, epoch, a=a_pred, u=u_pred, DE=DE, title='a', ks=KERNEL_SIZE, path=PATH)
	if loss_train < BEST_LOSS:
		torch.save(model.state_dict(), PATH + '/model.pt')
		BEST_LOSS = loss_train
	if np.isnan(loss_train):
		gc.collect()
		torch.cuda.empty_cache()
		raise Exception("Model diverged!")


time1 = time.time()
loss_plot(losses, FILE, EPOCHS, SHAPE, KERNEL_SIZE, BEST_LOSS, path=PATH)
dt = time1 - time0
AVG_ITER = np.round(dt/EPOCHS, 6)

params = {
	'MODEL': MODEL,
	'KERNEL_SIZE': KERNEL_SIZE,
	'FILE': FILE,
	'PATH': PATH,
	'BLOCKS': BLOCKS,
	'EPSILON': EPSILON,
	'FILTERS': FILTERS,
	'DATA': DATA,
	'EPOCHS': EPOCHS,
	'N': N,
	'LOSS': BEST_LOSS,
	'AVG_ITER': AVG_ITER
}

log_data(**params)

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()