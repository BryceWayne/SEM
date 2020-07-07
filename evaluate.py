#evaluate.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import LG_1d
from net.data_loader import *
from net.network import *
from sem.sem import *
from reconstruct import *
from plotting import *
import subprocess
import pandas as pd
import datetime


def validate(equation, model, optim, epsilon, shape, filters, criterion_a, criterion_u, criterion_f, criterion_wf, lepolys, phi, phi_x, phi_xx, A, U, F, WF):
	device = get_device()
	FILE, EQUATION, SHAPE, BATCH = f'1000N{shape-1}', equation, shape, 1000
	N, D_in, Filters, D_out = BATCH, 1, filters, shape
	# Load the dataset
	test_data = get_data(EQUATION, FILE, SHAPE, BATCH, D_out, epsilon, kind='validate')
	testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=False)
	loss = 0
	optim.zero_grad()
	for batch_idx, sample_batch in enumerate(testloader):
		f = sample_batch['f'].to(device)
		a = sample_batch['a'].to(device)
		u = sample_batch['u'].to(device)
		def closure(f, a, u):
			if torch.is_grad_enabled():
				optim.zero_grad()
			a_pred = model(f)
			loss_a = A*criterion_a(a_pred, a)
			if U != 0:
				u_pred = reconstruct(a_pred, phi)
				loss_u = U*criterion_u(u_pred, u)
			else:
				u_pred, loss_u = None, 0
			if F != 0:
				f_pred = ODE2(epsilon, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
				loss_f = F*criterion_f(f_pred, f)
			else:
				f_pred, loss_f = None, 0
			loss_wf = 0
			# LHS, RHS = weak_form2(epsilon, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION)
			# loss_wf = WF*criterion_wf(LHS, RHS)
			loss = loss_a + loss_u + loss_f + loss_wf
			return np.round(float(loss.to('cpu').detach()), 8)
		loss += closure(f, a, u)
	optim.zero_grad()
	return loss


def model_metrics(equation, input_model, file_name, ks, path, epsilon, filters, blocks):
	device = get_device()
	
	EQUATION, EPSILON, INPUT = equation, epsilon, file_name
	FILE = '1000N' + INPUT.split('N')[1]
	PATH = path
	KERNEL_SIZE = ks
	PADDING = (ks - 1)//2
	SHAPE, BATCH = int(FILE.split('N')[1]) + 1, 1000
	N, D_in, Filters, D_out = BATCH, 1, filters, SHAPE
	BLOCKS = blocks

	data = {}
	if input_model == ResNet:
		data['MODEL'] = 'ResNet'
	elif input_model == NetA:
		data['MODEL'] = 'NetA'
	title = data['MODEL']

	xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=equation)
	# LOAD MODEL
	model = input_model(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS).to(device)
	model.load_state_dict(torch.load(PATH + '/model.pt'))
	model.eval()

	if FILE.split('N')[1] != INPUT.split('N')[1]:
		FILE = '1000N' + INPUT.split('N')[1]
	test_data = get_data(EQUATION, FILE, SHAPE, BATCH, D_out, EPSILON, kind='validate')
	testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=False)

	running_MAE_a, running_MAE_u, running_MSE_a, running_MSE_u, running_MinfE_a, running_MinfE_u = 0, 0, 0, 0, 0, 0
	for batch_idx, sample_batch in enumerate(testloader):
		f = sample_batch['f'].to(device)
		u = sample_batch['u'].to(device)
		a = sample_batch['a'].to(device)
		a_pred = model(f)
		u_pred = reconstruct(a_pred, phi)
		f_pred = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
		a_pred = a_pred.to('cpu').detach().numpy()
		u_pred = u_pred.to('cpu').detach().numpy()
		f_pred = f_pred.to('cpu').detach().numpy()
		a = a.to('cpu').detach().numpy()
		u = u.to('cpu').detach().numpy()
		for i in range(N):
			running_MAE_a += mae(a_pred[i,0,:], a[i,0,:])
			running_MSE_a += relative_l2(a_pred[i,0,:], a[i,0,:])
			running_MinfE_a += relative_linf(a_pred[i,0,:], a[i,0,:])
			running_MAE_u += mae(u_pred[i,0,:], u[i,0,:])
			running_MSE_u += relative_l2(u_pred[i,0,:], u[i,0,:])
			running_MinfE_u += relative_linf(u_pred[i,0,:], u[i,0,:])

	out_of_sample(EQUATION, SHAPE, a_pred, u_pred, f_pred, sample_batch, PATH, title)
	
	data['EQUATION'] = equation
	data['TIMESTAMP'] = datetime.datetime.now()
	data['FOLDER'] = PATH[len(INPUT)+1:]
	try:
		data['FOLDER'] = data['FOLDER'].split('/')[1]
	except:	
		data['FOLDER'] = data['FOLDER'].split('\\')[1]
	data['DATASET'] = INPUT
	data['SHAPE'] = SHAPE
	data['K.SIZE'] = KERNEL_SIZE
	data['MAEa'] = np.round(running_MAE_a/N, 6)
	data['MSEa'] = np.round(running_MSE_a/N, 6)
	data['MIEa'] = np.round(running_MinfE_a/N, 6)
	data['MAEu'] = np.round(running_MAE_u/N, 6)
	data['MSEu'] = np.round(running_MSE_u/N, 6)
	data['MIEu'] = np.round(running_MinfE_u/N, 6)
	return data
