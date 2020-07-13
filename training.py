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
from tqdm import tqdm
import numpy as np
from net.data_loader import *
from net.network import *
from sem.sem import *
from plotting import *
from reconstruct import *
from data_logging import *
from evaluate import *


# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='Standard', choices=['Standard', 'Burgers', 'Helmholtz'])
parser.add_argument("--model", type=str, default='ResNet', choices=['ResNet', 'NetA']) 
parser.add_argument("--blocks", type=int, default=2)
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE'])
parser.add_argument("--file", type=str, default='500N63', help='Example: --file 2000N31')
parser.add_argument("--batch", type=int, default=500)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--nbfuncs", type=int, default=10, help='Number of basis functions to use in loss_wf')
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--transfer", type=str, default=None)
"""
Parabolic - Hyperbolic - Elliptic
MULTI-TASK LEARNING - learn generic feature extractor
MAML - model agnostic model learning 
"""
args = parser.parse_args()

#EQUATION
if args.equation == 'Standard':
	EPSILON = 1E-1
elif args.equation == 'Burgers':
	EPSILON = 5E-1
elif args.equation == 'Helmholtz':
	EPSILON = 0
EQUATION = args.equation

# MODEL
if args.model == 'ResNet':
	MODEL = ResNet
elif args.model == 'NetA':
	MODEL = NetA


#CREATE GLOBAL PARAMS
"""
global_params = {
	'EQUATION': args.equation,
	'EPSILON': EPSILON,
	'DATASET': args.file,
	'N': SHAPE,
	'TIME': cur_time,
	'PATH': PATH,
	'MODEL': args.model,
	'LOSS_TYPE': LOSS_TYPE,
	'BLOCKS': BLOCKS,
	'EPOCHS': EPOCHS,
	'KERNEL_SIZE': KERNEL_SIZE,
	'PADDING': PADDING,
	'FILTERS': FILTERS,
	'A': A,
	'U': U,
	'F': F,
	'WF': WF
}
"""

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
FOLDER = f'{args.model}_{args.loss}_epochs{args.epochs}_blocks{args.blocks}_nbfuncs{args.nbfuncs}'
PATH = os.path.join('training', f"{EQUATION}", FILE, FOLDER)
BLOCKS = args.blocks
NBFUNCS = args.nbfuncs

# #CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH)
elif os.path.isdir(PATH) == True: print("\n\nPATH ALREADY EXISTS!\n\n"); exit()
os.makedirs(os.path.join(PATH, 'pics'))

#CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)

lg_dataset = get_data(EQUATION, FILE, SHAPE, BATCH, SHAPE, EPSILON, kind='train')
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
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

"""
	ABLATION
	In the context of machine learning, we can define ablation study as,
	“a scientific examination of a machine learning system by removing its
		building blocks in order to gain insight on their effects on its overall 
		performance”. 
	Dataset features and model components are notable examples of these building blocks
	(hence we use their corresponding terms of feature ablation and model ablation), 
	but any design choice or module of the system may be included in an ablation study.
	SOURCE: https://www.quora.com/In-the-context-of-deep-learning-what-is-an-ablation-study
"""
"""
	(*) AMORITIZATION
	(*) Dropout, L2 Regularization on FC (Remedy for overfitting)
"""
A, U, F, WF = args.A, 1E3, 0E0, 1E3
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
			if WF != 0:
				LHS, RHS = weak_form2(EPSILON, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION, nbfuncs=NBFUNCS)
				loss_wf = WF*criterion_wf(LHS, RHS)
			else:
				loss_wf = 0
			loss = loss_a + loss_u + loss_f + loss_wf
			if loss.requires_grad:
				loss.backward()
			return a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss
		a_pred, u_pred, f_pred, loss_a, loss_u, loss_f, loss_wf, loss = closure(f, a, u)
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
		gc.collect()
		torch.cuda.empty_cache()
		raise Exception("Model diverged!")
	if loss_train/BATCH < BEST_LOSS:
		torch.save(model.state_dict(), PATH + '/model.pt')
		BEST_LOSS = loss_train/BATCH

	loss_validate = validate(EQUATION, model, optimizer, EPSILON, SHAPE, FILTERS, criterion_a, criterion_u, criterion_f, criterion_wf, lepolys, phi, phi_x, phi_xx, A, U, F, WF, NBFUNCS)
	if loss_a != 0:
		losses['loss_a'].append(loss_a.item()/BATCH)
	else:
		losses['loss_a'].append(loss_a/BATCH)
	if loss_u != 0:
		losses['loss_u'].append(loss_u.item()/BATCH)
	else:
		losses['loss_u'].append(loss_u/BATCH)
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

	if int(.05*EPOCHS) > 0 and EPOCHS > 10 and epoch % int(.05*EPOCHS) == 0:
		print(f"\nT. Loss: {np.round(losses['loss_train'][-1], 9)}, "\
			  f"V. Loss: {np.round(losses['loss_validate'][-1], 9)}")
		f_pred = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
		# f_pred = None
		plotter(xx, sample_batch, epoch, a=a_pred, u=u_pred, f=f_pred, title=args.model, ks=KERNEL_SIZE, path=PATH)

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

df = log_data(**params)
loss_log(params, losses, df)

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
