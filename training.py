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
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from net.data_loader import *
from net.network import *
from sem.sem import *
from plotting import *
from reconstruct import *
from data_logging import *
from evaluate import *
from pprint import pprint


# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Burgers', choices=['Standard', 'Burgers', 'Helmholtz', 'BurgersT']) # 
parser.add_argument("--model", type=str, default='NetA', choices=['ResNet', 'NetA', 'NetB', 'NetC']) # , 'Net2D' 
parser.add_argument("--blocks", type=int, default=2)
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE'])
parser.add_argument("--file", type=str, default='75000N31', help='Example: --file 2000N31')
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--ks", type=int, default=5, choices=[3, 5, 7, 9, 11, 13, 15, 17])
parser.add_argument("--filters", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--nbfuncs", type=int, default=1, help='Number of basis functions to use in loss_wf')
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--F", type=float, default=1E3)
parser.add_argument("--transfer", type=str, default=None)

args = parser.parse_args()
gparams = args.__dict__

EQUATION = args.equation
epsilons = {'Standard': 1E-1,
			'Burgers': 1,
			'BurgersT': 1,
			'Helmholtz': 0,
			}
EPSILON = epsilons[EQUATION]
models = {'ResNet': ResNet,
		  'NetA': NetA,
		  'NetB': NetB,
		  'NetC': NetC,
		  'Net2D': Net2D,
		  }
MODEL = models[args.model]

#GLOBALS
FILE = args.file
DATASET = int(args.file.split('N')[0])
SHAPE = int(args.file.split('N')[1]) + 1
BLOCKS = args.blocks
EPOCHS = args.epochs
NBFUNCS = args.nbfuncs
FILTERS = args.filters
KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{args.model}_epochs{EPOCHS}_{cur_time}'
PATH = os.path.join('training', f"{EQUATION}", FILE, FOLDER)
BATCH_SIZE, D_in, Filters, D_out = DATASET, 1, FILTERS, SHAPE
# LOSS SCALE FACTORS
A, U, F, WF = args.A, 1E3, args.F, 1E3

gparams['U'] = U
gparams['WF'] = WF
gparams['epsilon'] = EPSILON

# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH); os.makedirs(os.path.join(PATH, 'pics'))
elif os.path.isdir(PATH) == True:
	if args.transfer is None:
		print("\n\nPATH ALREADY EXISTS!\n\nEXITING\n\n")
		exit()
	elif args.transfer is not None:
		print("\n\nPATH ALREADY EXISTS!\n\nLOADING MODEL\n\n")

gparams['path'] = PATH
with open("paths.txt", "a") as f:
	f.write(str(PATH) + '\n')

# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)

# LOAD DATASET
# transform_f = 
lg_dataset = get_data(EQUATION, FILE, SHAPE, DATASET, EPSILON, kind='train')
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = MODEL(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
if args.transfer is not None:
	model.load_state_dict(torch.load(f'./{args.transfer}.pt'))
	model.train()	

#KAIMING HE INIT
if args.transfer is None:
	model.apply(weights_init)
# Check if CUDA is available and then use it.
device = get_device()
# SEND TO GPU (or CPU)
model.to(device)
#INIT OPTIMIZER
optimizer = init_optim(model)

# Construct our loss function and an Optimizer.
LOSS_TYPE = args.loss
if args.loss == 'MAE':
	criterion_a, criterion_u = torch.nn.L1Loss(), torch.nn.L1Loss()
elif args.loss == 'MSE':
	criterion_a, criterion_u = torch.nn.MSELoss(reduction="sum"), torch.nn.MSELoss(reduction="sum")
elif args.loss == 'RMSE':
	criterion_a, criterion_u = RMSELoss(), RMSELoss()
criterion_wf = torch.nn.MSELoss(reduction="sum")
criterion_f = torch.nn.L1Loss()


BEST_LOSS = float('inf')
losses = {'loss_a':[],
		  'loss_u':[], 
		  'loss_f': [], 
		  'loss_wf':[], 
		  'loss_train':[], 
		  'loss_validate':[]
		  }

log_gparams(gparams)

time0 = time.time()
for epoch in tqdm(range(1, EPOCHS+1)):
	loss_a, loss_u, loss_f, loss_wf, loss_train = 0, 0, 0, 0, 0
	for batch_idx, sample_batch in enumerate(trainloader):
		f = sample_batch['f'].to(device)
		a = sample_batch['a'].to(device)
		u = sample_batch['u'].to(device)
		def closure(a, f, u):
			if torch.is_grad_enabled():
				optimizer.zero_grad()
			a_pred = model(f)
			if A != 0:
				loss_a = A*criterion_a(a_pred, a)
			else:
				loss_a = 0
			if U != 0:
				u_pred = reconstruct(a_pred, phi)
				loss_u = U*criterion_u(u_pred, u)
			else:
				u_pred, loss_u = None, 0
			if F != 0:
				f_pred = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
				loss_f = F*criterion_f(f_pred, f)
			else:
				f_pred, loss_f = None, 0
			if WF != 0 and EQUATION != 'BurgersT':
				LHS, RHS = weak_form2(EPSILON, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION, nbfuncs=NBFUNCS)
				loss_wf = WF*criterion_wf(LHS, RHS)
			else:
				loss_wf = 0
			# NET LOSS
			loss = loss_a + loss_u + loss_f + loss_wf
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss

		a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss = closure(a, f, u)
		optimizer.step(loss.item)
		if loss_a != 0:
			loss_a += np.round(float(loss_a.to('cpu').detach()), 9)
		if loss_u != 0:
			loss_u += np.round(float(loss_u.to('cpu').detach()), 9)
		if loss_f != 0:
			loss_f += np.round(float(loss_f.to('cpu').detach()), 9)
		if loss_wf != 0:
			loss_wf += np.round(float(loss_wf.to('cpu').detach()), 9)
		loss_train += np.round(float(loss.to('cpu').detach()), 9)

	if np.isnan(loss_train):
		try:
			model.load_state_dict(torch.load(PATH + '/model.pt'))
			model.train()
			optimizer = init_optim(model)
			print('Model diverged! Optimizer reinitialized')
		except:
			raise Exception("Model diverged! Unable to load a previous state.")
	else:
		if loss_train/DATASET < BEST_LOSS:
			torch.save(model.state_dict(), PATH + '/model.pt')
			BEST_LOSS = loss_train/DATASET
			gparams['best_loss'] = BEST_LOSS
			
		loss_validate = validate(EQUATION, model, optimizer, EPSILON, SHAPE, FILTERS, criterion_a, criterion_u, criterion_f, criterion_wf, lepolys, phi, phi_x, phi_xx, A, U, F, WF, NBFUNCS)
		losses = log_loss(losses, loss_a, loss_u, loss_f, loss_wf, loss_train, loss_validate, DATASET)
		
		if int(.05*EPOCHS) > 0 and EPOCHS > 10 and epoch % int(.05*EPOCHS) == 0:
			periodic_report(args.model, sample_batch, EQUATION, EPSILON, SHAPE, epoch, xx, phi_x, phi_xx, losses, a_pred, u_pred, f_pred, KERNEL_SIZE, PATH)

time1 = time.time()
dt = time1 - time0
AVG_ITER = np.round(dt/EPOCHS, 6)
NPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

gparams['dt'] = dt
gparams['AvgIter'] = AVG_ITER
gparams['nparams'] = NPARAMS

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
	'BATCH_SIZE': BATCH_SIZE,
	'LOSS': BEST_LOSS,
	'AVG_ITER': AVG_ITER,
	'LOSSES': losses,
	'LOSS_TYPE': LOSS_TYPE,
	'NBFUNCS': NBFUNCS,
	'NPARAMS': NPARAMS
}

gparams['losses'] = losses
loss_plot(losses, FILE, EPOCHS, SHAPE, KERNEL_SIZE, BEST_LOSS, PATH, title=args.model)
df = log_data(**params)
loss_log(params, losses ,df)
log_gparams(gparams)
model_stats(PATH)

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
