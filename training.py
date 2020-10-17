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
parser.add_argument("--equation", type=str, default='Burgers', choices=['Standard', 'Burgers', 'Helmholtz']) #, 'BurgersT' 
parser.add_argument("--model", type=str, default='NetC', choices=['ResNet', 'NetA', 'NetB', 'NetC', 'NetD']) # , 'Net2D' 
parser.add_argument("--blocks", type=int, default=4)
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--file", type=str, default='10000N63', help='Example: --file 2000N31')
parser.add_argument("--forcing", type=str, default='uniform', choices=['normal', 'uniform'])
parser.add_argument("--epochs", type=int, default=50000)
parser.add_argument("--ks", type=int, default=5, choices=[3, 5, 7, 9, 11, 13, 15, 17])
parser.add_argument("--filters", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--nbfuncs", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--F", type=float, default=0)
parser.add_argument("--U", type=float, default=1)
parser.add_argument("--WF", type=float, default=1)
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--transfer", type=str, default=None)

args = parser.parse_args()
gparams = args.__dict__
pprint(gparams)

EQUATION = args.equation
epsilons = {
			'Standard': 1E-1,
			'Burgers': 5E-1,
			'BurgersT': 1,
			'Helmholtz': 0,
			}
EPSILON = epsilons[EQUATION]
models = {
		  'ResNet': ResNet,
		  'NetA': NetA,
		  'NetB': NetB,
		  'NetC': NetC,
		  'NetD': NetD,
		  'Net2D': Net2D,
		  }
MODEL = models[args.model]

#GLOBALS
FILE = gparams['file']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1
BLOCKS = int(gparams['blocks'])
EPOCHS = int(gparams['epochs'])
NBFUNCS = int(gparams['nbfuncs'])
FILTERS = int(gparams['filters'])
KERNEL_SIZE = int(gparams['ks'])
PADDING = (KERNEL_SIZE - 1)//2
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{gparams["model"]}_epochs{EPOCHS}_{cur_time}'
PATH = os.path.join('training', f"{EQUATION}", FILE, FOLDER)
gparams['path'] = PATH
BATCH_SIZE, D_in, Filters, D_out = DATASET, 1, FILTERS, SHAPE
# LOSS SCALE FACTORS
A, U, F, WF = int(gparams['A']), int(gparams['U']), int(gparams['F']), int(gparams['WF'])

gparams['epsilon'] = EPSILON

# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH); os.makedirs(os.path.join(PATH, 'pics'))
elif os.path.isdir(PATH) == True:
	if args.transfer is None:
		print("\n\nPATH ALREADY EXISTS!\n\nEXITING\n\n")
		exit()
	elif args.transfer is not None:
		print("\n\nPATH ALREADY EXISTS!\n\nLOADING MODEL\n\n")

# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)

if args.model != 'ResNet':
	# NORMALIZE DATASET
	NORM = True
	gparams['norm'] = True
	lg_dataset = get_data(gparams, kind='train')
	trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)
	gparams, transform_f = normalize(gparams, trainloader)
else:
	NORM = False
	gparams['norm'] = False
	transform_f = None
# LOAD DATASET
lg_dataset = get_data(gparams, kind='train', transform_f=transform_f)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)
lg_dataset = get_data(gparams, kind='validate', transform_f=transform_f)
validateloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

if args.model == 'FC':
	model = MODEL(D_in, Filters, D_out - 2, layers=BLOCKS, activation='relu')
else:
	model = MODEL(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
if args.transfer is not None:
	model.load_state_dict(torch.load(f'./{args.transfer}.pt'))
	model.train()	

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device
# SEND TO GPU (or CPU)
model.to(device)
#KAIMING HE INIT
if args.transfer is None and args.model != 'NetB':
	model.apply(weights_init)
elif args.model == 'NetB':
	model.apply(weights_xavier)

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
elif args.loss == 'RelMSE':
	criterion_a, criterion_u = RelMSELoss(batch=BATCH_SIZE), RelMSELoss(batch=BATCH_SIZE)
criterion_wf = torch.nn.MSELoss(reduction="sum")
# criterion_wf = torch.nn.L1Loss()
criterion_f = torch.nn.L1Loss()

criterion = {
			 'a': criterion_a,
			 'f': criterion_f,
			 'u': criterion_u,
			 'wf': criterion_wf,
			}
BEST_LOSS = float('inf')
losses = {
		  'loss_a':[],
		  'loss_u':[], 
		  'loss_f': [], 
		  'loss_wf1':[],
		  'loss_wf2':[],
		  'loss_wf3':[],
		  'loss_train':[], 
		  'loss_validate':[],
		  'avg_l2_u': []
		  }

log_gparams(gparams)

time0 = time.time()
for epoch in tqdm(range(1, EPOCHS+1)):
	loss_a, loss_u, loss_f, loss_wf, loss_train = 0, 0, 0, 0, 0
	for batch_idx, sample_batch in enumerate(trainloader):
		f = sample_batch['f'].to(device)
		if NORM is False:
			fn = f
		elif NORM == True:
			fn = sample_batch['fn'].to(device)
		a = sample_batch['a'].to(device)
		u = sample_batch['u'].to(device)
		def closure(a, f, u, fn=fn):
			if torch.is_grad_enabled():
				optimizer.zero_grad()
			a_pred = model(fn)
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
				# print("Nbfuncs:", NBFUNCS, "\nLHS:", LHS.shape, "\nRHS:",  RHS.shape)
				if NBFUNCS == 1:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS, RHS), 0, 0
				elif NBFUNCS == 2:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS[:,0,0], RHS[:,0,0]), WF*criterion_wf(LHS[:,1,0], RHS[:,1,0]), 0
				elif NBFUNCS == 3:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS[:,0,0], RHS[:,0,0]), WF*criterion_wf(LHS[:,1,0], RHS[:,1,0]), WF*criterion_wf(LHS[:,2,0], RHS[:,2,0])
			else:
				loss_wf1, loss_wf2, loss_wf3 = 0, 0, 0
			# NET LOSS
			loss = loss_a + loss_u + loss_f + loss_wf1 + loss_wf2 + loss_wf3
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf1, loss_wf2, loss_wf3, loss

		a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf1, loss_wf2, loss_wf3, loss = closure(a, f, u, fn)
		optimizer.step(loss.item)
		if loss_a != 0:
			loss_a += np.round(float(loss_a.to('cpu').detach()), 12)
		if loss_u != 0:
			loss_u += np.round(float(loss_u.to('cpu').detach()), 12)
		if loss_f != 0:
			loss_f += np.round(float(loss_f.to('cpu').detach()), 12)
		if loss_wf1 != 0:
			loss_wf1 += np.round(float(loss_wf1.to('cpu').detach()), 12)
		else:
			loss_wf1 += 0
		if loss_wf2 > 0:
			loss_wf2 += np.round(float(loss_wf2.to('cpu').detach()), 12)
		else:
			loss_wf2 += 0	
		if loss_wf3 > 0:
			loss_wf3 += np.round(float(loss_wf3.to('cpu').detach()), 12)
		else:
			loss_wf3 += 0
		loss_train += np.round(float(loss.to('cpu').detach()), 12)

	if np.isnan(loss_train):
		try:
			model.load_state_dict(torch.load(PATH + '/model.pt'))
			model.train()
			optimizer = init_optim(model)
			print('Model diverged! Optimizer reinitialized')
		except:
			model = MODEL(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
			model.to(device)
			optimizer = init_optim(model)
			print('Model diverged! Model & Optimizer reinitialized')
		finally:
			raise Exception("Model diverged! Unable to load a previous state.")
	else:
		if loss_train/DATASET < BEST_LOSS:
			torch.save(model.state_dict(), PATH + '/model.pt')
			BEST_LOSS = loss_train/DATASET
			gparams['best_loss'] = BEST_LOSS

		avg_l2_u, loss_validate = validate(gparams, model, optimizer, criterion, lepolys, phi, phi_x, phi_xx, validateloader)
		losses = log_loss(losses, loss_a, loss_u, loss_f, loss_wf1, loss_wf2, loss_wf3, loss_train, loss_validate, BATCH_SIZE, avg_l2_u)

		if int(.05*EPOCHS) > 0 and EPOCHS > 10 and epoch % int(.05*EPOCHS) == 0:
			try:
				df = pd.DataFrame(losses)
				df.to_csv(PATH + '/losses.csv')
				del df
			except:
				print('Unable to save losses.')
			periodic_report(args.model, sample_batch, EQUATION, EPSILON, SHAPE, epoch, xx, phi_x, phi_xx, losses, a_pred, u_pred, f_pred, KERNEL_SIZE, PATH)

time1 = time.time()
dt = time1 - time0
AVG_ITER = np.round(dt/EPOCHS, 6)
NPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

gparams['dt'] = dt
gparams['avgIter'] = AVG_ITER
gparams['nParams'] = NPARAMS
gparams['batchSize'] = BATCH_SIZE
gparams['bestLoss'] = BEST_LOSS
gparams['losses'] = losses
gparams['lossType'] = LOSS_TYPE

log_path(PATH)

loss_plot(gparams)
values = model_stats(PATH, kind='validate', gparams=gparams)
for k, v in values.items():
	gparams[k] = np.mean(v)

log_gparams(gparams)

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
