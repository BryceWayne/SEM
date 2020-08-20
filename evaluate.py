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


def validate(equation, model, optim, epsilon, shape, filters, criterion_a, criterion_u, criterion_f, criterion_wf, lepolys, phi, phi_x, phi_xx, A, U, F, WF, nbfuncs):
	device = get_device()
	FILE, EQUATION, SHAPE = f'10000N{shape-1}', equation, shape
	BATCH_SIZE, D_in, Filters, D_out = 10000, 1, filters, shape
	# Load the dataset
	test_data = get_data(EQUATION, FILE, SHAPE, BATCH_SIZE, epsilon, kind='validate')
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
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
				f_pred = ODE2(epsilon, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
				loss_f = F*criterion_f(f_pred, f)
			else:
				f_pred, loss_f = None, 0
			if WF != 0 and equation != 'BurgersT':
				LHS, RHS = weak_form2(epsilon, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION, nbfuncs=nbfuncs)
				loss_wf = WF*criterion_wf(LHS, RHS)
			else:
				loss_wf = 0
			loss = loss_a + loss_u + loss_f + loss_wf
			return np.round(float(loss.to('cpu').detach()), 8)
		loss += closure(f, a, u)
	optim.zero_grad()
	return loss


def model_metrics(equation, input_model, file_name, ks, path, epsilon, filters, blocks):
	device = get_device()
	
	EQUATION, EPSILON, INPUT = equation, epsilon, file_name
	FILE = '10000N' + INPUT.split('N')[1]
	PATH = path
	KERNEL_SIZE = ks
	PADDING = (ks - 1)//2
	SHAPE = int(FILE.split('N')[1]) + 1
	BATCH_SIZE, D_in, Filters, D_out = 10000, 1, filters, SHAPE
	BLOCKS = blocks

	data = {}
	if input_model == ResNet:
		data['MODEL'] = 'ResNet'
	elif input_model == NetA:
		data['MODEL'] = 'NetA'
	elif input_model == NetB:
		data['MODEL'] = 'NetB'
	elif input_model == NetC:
		data['MODEL'] = 'NetC'
	elif input_model == Net2D:
		data['MODEL'] = 'Net2D'
		
	title = data['MODEL']

	xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)
	# LOAD MODEL
	model = input_model(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS).to(device)
	model.load_state_dict(torch.load(PATH + '/model.pt'))
	model.eval()

	if FILE.split('N')[1] != INPUT.split('N')[1]:
		FILE = '10000N' + INPUT.split('N')[1]
	test_data = get_data(EQUATION, FILE, SHAPE, BATCH_SIZE, EPSILON, kind='validate')
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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
		for i in range(BATCH_SIZE):
			running_MAE_a += mae(a_pred[i,0,:], a[i,0,:])
			running_MSE_a += relative_l2(a_pred[i,0,:], a[i,0,:])
			running_MinfE_a += linf(a_pred[i,0,:], a[i,0,:])
			running_MAE_u += mae(u_pred[i,0,:], u[i,0,:])
			running_MSE_u += relative_l2(u_pred[i,0,:], u[i,0,:])
			running_MinfE_u += linf(u_pred[i,0,:], u[i,0,:])

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
	data['MAEa'] = np.round(running_MAE_a/BATCH_SIZE, 6)
	data['MSEa'] = np.round(running_MSE_a/BATCH_SIZE, 6)
	data['MIEa'] = np.round(running_MinfE_a/BATCH_SIZE, 6)
	data['MAEu'] = np.round(running_MAE_u/BATCH_SIZE, 6)
	data['MSEu'] = np.round(running_MSE_u/BATCH_SIZE, 6)
	data['MIEu'] = np.round(running_MinfE_u/BATCH_SIZE, 6)
	return data


def model_stats(path):
	# def model_metrics(equation, input_model, file_name, ks, path, epsilon, filters, blocks):
	device = get_device()
	cwd = os.getcwd()
	os.chdir(path)
	with open("parameters.txt", 'r') as f:
		text = f.readlines()
	from pprint import pprint
	os.chdir(cwd)
	# pprint(text)
	for i, _ in enumerate(text):
		text[i] = _.rstrip('\n')
	gparams = {}
	for i, _ in enumerate(text):
		_ = _.split(':')
		k, v = _[0], _[1]
		try:
			gparams[k] = float(v)
		except:
			gparams[k] = v
	pprint(gparams)
	# print(gparams['model'])
	if gparams['model'] == 'ResNet':
		model = ResNet
	elif gparams['model'] == 'NetA':
		model = NetA
	elif gparams['model'] == 'NetB':
		model = NetB
	elif gparams['model'] == 'NetC':
		model = NetC
	
	EQUATION, EPSILON, INPUT = gparams['equation'], gparams['epsilon'], gparams['file']
	FILE = '10000N' + INPUT.split('N')[1]
	PATH = gparams['path']
	KERNEL_SIZE = int(gparams['ks'])
	PADDING = (KERNEL_SIZE - 1)//2
	SHAPE = int(FILE.split('N')[1]) + 1
	BATCH_SIZE, D_in, Filters, D_out = 10000, 1, int(gparams['filters']), SHAPE
	BLOCKS = int(gparams['blocks'])

	xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)
	# # LOAD MODEL
	model = model(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS).to(device)
	model.load_state_dict(torch.load(PATH + '/model.pt'))
	model.eval()

	test_data = get_data(EQUATION, FILE, SHAPE, BATCH_SIZE, EPSILON, kind='validate')
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

	MAE_a, MSE_a, MinfE_a, MAE_u, MSE_u, MinfE_u, pwe_a, pwe_u = [], [], [], [], [], [], [], []
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
		for i in range(BATCH_SIZE):
			MAE_a.append(mae(a_pred[i,0,:], a[i,0,:]))
			MSE_a.append(relative_l2(a_pred[i,0,:], a[i,0,:]))
			MinfE_a.append(linf(a_pred[i,0,:], a[i,0,:]))
			MAE_u.append(mae(u_pred[i,0,:], u[i,0,:]))
			MSE_u.append(relative_l2(u_pred[i,0,:], u[i,0,:]))
			MinfE_u.append(linf(u_pred[i,0,:], u[i,0,:]))
			pwe_a.append(np.round(a_pred[i,0,:] - a[i,0,:], 9))
			pwe_u.append(np.round(u_pred[i,0,:] - u[i,0,:], 9))
			if relative_l2(u_pred[i,0,:], u[i,0,:]) > 1:
				plt.plot(xx, u_pred[i,0,:], label='Pred')
				plt.plot(xx, u[i,0,:], label='True')
				plt.legend()
				plt.xlim(-1, 1)
				plt.show()
	
	values = {
		'MAE_a': MAE_a,
		'MSE_a': MSE_a,
		'MinfE_a': MinfE_a,
		'MAE_u': MAE_u,
		'MSE_u': MSE_u,
		'MinfE_u': MinfE_u,
		'PWE_a': pwe_a,
		'PWE_u': pwe_u
	}

	df = pd.DataFrame(values)
	os.chdir(path)
	df.to_csv('out_of_sample.csv')
	try:
		df = pd.DataFrame(gparams['losses'])
		df.to_csv('losses.csv')
	except:
		pass
	os.chdir(cwd)
	return values