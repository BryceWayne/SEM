#training.py
import random
import torch
import time
import datetime
import subprocess
import os
import LG_1d
import argparse
import gc
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.sparse import diags
from tqdm import tqdm
import numpy as np
import scipy as sp
import pandas as pd
from net.data_loader import *
from net.network import *
from sem.sem import *
from plotting import *
from reconstruct import *
from data_logging import *
from evaluate_a import *


# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()


# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--model", type=str, default='NetA', choices=['ResNet', 'NetA']) 
parser.add_argument("--equation", type=str, default='Burgers', choices=['Standard', 'Burgers'])
parser.add_argument('--eps', type=float, default=1E-1)
parser.add_argument("--loss", type=str, default='MAE', choices=['MAE', 'MSE'])
parser.add_argument("--file", type=str, default='100000N31', help='Example: --file 2000N31')
parser.add_argument("--batch", type=int, default=10000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--filters", type=int, default=32)
args = parser.parse_args()


# VARIABLES
if args.model == 'ResNet':
	MODEL = ResNet
elif args.model == 'NetA':
	MODEL = NetA


EQUATION = args.equation
FILE = args.file
BATCH = int(args.file.split('N')[0])
SHAPE = int(args.file.split('N')[1]) + 1
FILTERS = args.filters
KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
EPOCHS = args.epochs
N, D_in, Filters, D_out = args.batch, 1, FILTERS, SHAPE
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
PATH = os.path.join(FILE, f"{EQUATION}", cur_time)
BLOCKS = args.blocks
EPSILON = args.eps


# #CREATE PATHING
# if os.path.isdir(FILE) == False:
# 	os.makedir(FILE)
# try:
# 	os.mkdir(os.path.join(FILE, EQUATION))
# except:
# 	pass
if os.path.isdir(os.path.join(FILE, EQUATION)) == False:
	os.makedirs(os.path.join(FILE, EQUATION))
os.makedirs(os.path.join(PATH,'pics'))

#CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out)

lg_dataset = get_data(EQUATION, FILE, SHAPE, BATCH, D_out, EPSILON)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
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
if args.loss == 'MAE':
	LOSS_TYPE = args.loss
	criterion_a = torch.nn.L1Loss()
	criterion_u = torch.nn.L1Loss()
	criterion_f = torch.nn.L1Loss()
	criterion_wf = torch.nn.L1Loss()
elif args.loss == 'MSE':
	LOSS_TYPE = args.loss
	criterion_a = torch.nn.MSELoss(reduction="sum")
	criterion_u = torch.nn.MSELoss(reduction="sum")
	criterion_f = torch.nn.MSELoss(reduction="sum")
	criterion_wf = torch.nn.MSELoss(reduction="sum")

optimizer = torch.optim.LBFGS(model.parameters(), history_size=20, tolerance_grad=1e-15, tolerance_change=1e-15, max_eval=20)
# optimizer = torch.optim.SGD(model.parameters(), lr=1E-8)

"""
	(*) AMORITIZATION
	(*) Dropout, L2 Regularization on FC (Remedy for overfitting)
"""
BEST_LOSS, losses = float('inf'), {'loss_a':[], 'loss_u':[], 'loss_f': [], 'loss_wf':[], 'loss_train':[], 'loss_validate':[]}
time0 = time.time()
for epoch in tqdm(range(1, EPOCHS+1)):
	loss_a, loss_u, loss_f, loss_wf, loss_train = 0, 0, 0, 0, 0
	for batch_idx, sample_batch in enumerate(trainloader):
		f = sample_batch['f'].to(device)
		a = sample_batch['a'].to(device)
		u = sample_batch['u'].to(device)
		def closure(f, a, u):
			if torch.is_grad_enabled():
				optimizer.zero_grad()
			a_pred = model(f)
			assert a_pred.shape == a.shape
			loss_a = criterion_a(a_pred, a)
			u_pred = reconstruct(a_pred, phi)
			assert u_pred.shape == u.shape
			loss_u = criterion_u(u_pred, u)
			f_pred, loss_f = None, 0
			LHS, RHS, loss_wf = 0, 0, 0
			# LHS, RHS = weak_form2(EPSILON, SHAPE, f, u, a_pred, lepolys, phi, phi_x, equation=EQUATION)
			# loss_wf = 1E1*criterion_wf(LHS, RHS)
			# loss_wf = 0
			loss = loss_a + loss_u + loss_f + loss_wf	
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss
		a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss = closure(f, a, u)
		optimizer.step(loss.item)
		if loss_a != 0:
			loss_a += np.round(float(loss_a.to('cpu').detach()), 8)
		loss_u += np.round(float(loss_u.to('cpu').detach()), 8)
		if loss_f != 0:
			loss_f += np.round(float(loss_f.to('cpu').detach()), 8)
		if loss_wf != 0:
			loss_wf += np.round(float(loss_wf.to('cpu').detach()), 8)
		loss_train += np.round(float(loss.to('cpu').detach()), 8)
	loss_validate = validate(EQUATION, model, optimizer, EPSILON, SHAPE, FILTERS, criterion_a, criterion_u, criterion_f, criterion_wf, lepolys, phi, phi_x, phi_xx)
	losses['loss_a'].append(loss_a.item()/BATCH)
	losses['loss_u'].append(loss_u.item()/BATCH)
	if type(loss_f) == int:
		losses['loss_f'].append(loss_f/BATCH) 
	else:
		losses['loss_f'].append(loss_f.item()/BATCH)
	if type(loss_wf) == int:
		losses['loss_wf'].append(loss_wf/BATCH) 
	else:
		losses['loss_wf'].append(loss_wf.item()/BATCH)
	losses['loss_train'].append(loss_train.item()/BATCH)
	losses['loss_validate'].append(loss_validate.item()/1000)

	if EPOCHS > 10 and epoch % int(.05*EPOCHS) == 0:
		print(f"T. Loss: {np.round(losses['loss_train'][-1], 9)}, "\
			  f"V. Loss: {np.round(losses['loss_validate'][-1], 9)}")
		f_pred = ODE2(EPSILON, u, a_pred, phi_x, phi_xx, equation=EQUATION)
		plotter(xx, sample_batch, epoch, a=a_pred, u=u_pred, DE=f_pred, title=args.model, ks=KERNEL_SIZE, path=PATH)
	if loss_train < BEST_LOSS:
		torch.save(model.state_dict(), PATH + '/model.pt')
		BEST_LOSS = loss_train
	if np.isnan(loss_train):
		gc.collect()
		torch.cuda.empty_cache()
		raise Exception("Model diverged!")


time1 = time.time()
loss_plot(losses, FILE, EPOCHS, SHAPE, KERNEL_SIZE, BEST_LOSS, PATH, title=args.model)
dt = time1 - time0
AVG_ITER = np.round(dt/EPOCHS, 6)

params = {
	'EQUATION': EQUATION,
	'MODEL': MODEL,
	'KERNEL_SIZE': KERNEL_SIZE,
	'FILE': FILE,
	'PATH': PATH,
	'BLOCKS': BLOCKS,
	'EPSILON': EPSILON,
	'FILTERS': FILTERS,
	'EPOCHS': EPOCHS,
	'N': N,
	'LOSS': BEST_LOSS,
	'AVG_ITER': AVG_ITER,
	'LOSSES': losses,
	'LOSS_TYPE': LOSS_TYPE
}

log_data(**params)

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()