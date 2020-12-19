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
from data_logging import *
import subprocess
import pandas as pd
import datetime
import numpy as np
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
import seaborn as sns



def validate(gparams, model, optim, criterion, lepolys, phi, phi_x, phi_xx, validateloader):
	device = gparams['device']
	VAL_SIZE = 1000
	NORM = gparams['norm']
	if NORM == 'False':
		NORM = False
	elif NORM == 'True':
		NORM = True
	SHAPE, EPSILON =  int(gparams['file'].split('N')[1]) + 1, gparams['epsilon']
	FILE, EQUATION = f'{VAL_SIZE}N{SHAPE-1}', gparams['equation']
	BATCH_SIZE, D_in, Filters, D_out = VAL_SIZE, 1, gparams['filters'], SHAPE
	NBFUNCS, SD = gparams['nbfuncs'], gparams['sd']
	A, F, U, WF = gparams['A'], gparams['F'], gparams['U'], gparams['WF']
	criterion_a, criterion_u = criterion['a'], criterion['u']
	criterion_f, criterion_wf = criterion['f'], criterion['wf']
	forcing = gparams['forcing']
	loss = 0
	optim.zero_grad()
	for batch_idx, sample_batch in enumerate(validateloader):
		f = sample_batch['f'].to(device)
		if NORM == True:
			fn = sample_batch['fn'].to(device)
		else:
			fn = sample_batch['f'].to(device)
		a = sample_batch['a'].to(device)
		u = sample_batch['u'].to(device)
		def closure(f, a, u, fn=fn):
			if torch.is_grad_enabled():
				optim.zero_grad()
			a_pred = model(fn)
			loss_a = 0
			u_pred = reconstruct(a_pred, phi)
			# if u_pred.shape[1] == 2: # B, 2, N+1
			# 	T1 = u_pred[:,0:1,:].permute(0,2,1)
			# 	T1 = T1.reshape(BATCH_SIZE, SHAPE, 1)
			# 	T2 = u_pred[:,1:2,:]
			# 	T2 = T2.reshape(BATCH_SIZE, 1, SHAPE)
			# 	uu = torch.bmm(T1, T2)
			# 	uu = uu.reshape(BATCH_SIZE, 1,  SHAPE, SHAPE)
			# 	u = u.reshape(BATCH_SIZE, 1,  SHAPE, SHAPE)
			# else:
			# 	uu = None
			loss_u = U*criterion_u(u_pred, u)
			f_pred, loss_f = None, 0
			if WF != 0:
				LHS, RHS = weak_form2(EPSILON, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION, nbfuncs=NBFUNCS)
				if NBFUNCS == 1:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS, RHS), 0, 0
				elif NBFUNCS == 2:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS[:,0,0], RHS[:,0,0]), WF*criterion_wf(LHS[:,1,0], RHS[:,1,0]), 0
				elif NBFUNCS == 3:
					loss_wf1, loss_wf2, loss_wf3 = WF*criterion_wf(LHS[:,0,0], RHS[:,0,0]), WF*criterion_wf(LHS[:,1,0], RHS[:,1,0]), WF*criterion_wf(LHS[:,2,0], RHS[:,2,0])
			else:
				loss_wf1, loss_wf2, loss_wf3 = 0, 0, 0
			loss = loss_a + loss_u + loss_f + loss_wf1 + loss_wf2 + loss_wf3
			u_pred = u_pred.to('cpu').detach().numpy()
			u = u.to('cpu').detach().numpy()
			avg_l2_u = 0
			for i in range(BATCH_SIZE):
				if len(u_pred) == 3:
					avg_l2_u += relative_l2(u_pred[i,0,:], u[i,0,:])
				else:
					# print(i, avg_l2_u)
					# avg_l2_u += relative_l2(u_pred[i,0,:,:], u[i,0,:,:])
					avg_l2_u = 0
			avg_l2_u /= BATCH_SIZE
			return avg_l2_u, np.round(float(loss.item()), 12)
		avg_l2_u, loss = closure(f, a, u, fn)
	optim.zero_grad()
	return avg_l2_u, loss


def model_stats(path, kind='train', gparams=None):
	from torchsummary import summary
	red, blue, green, purple = color_scheme()
	TEST  = {'color':red, 'marker':'o', 'linestyle':'none', 'markersize': 3}
	VAL = {'color':blue, 'marker':'o', 'linestyle':'solid', 'mfc':'none'}
	cwd = os.getcwd()
	if gparams == None:
		os.chdir(path)
		with open("parameters.txt", 'r') as f:
			text = f.readlines()
			f.close()
		from pprint import pprint
		os.chdir(cwd)

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

	if gparams['model'] == 'ResNet':
		model = ResNet
	elif gparams['model'] == 'NetA':
		model = NetA
	elif gparams['model'] == 'NetB':
		model = NetB
	elif gparams['model'] == 'NetC':
		model = NetC
	elif gparams['model'] == 'NetD':
		model = NetD

	EQUATION, EPSILON, INPUT = gparams['equation'], gparams['epsilon'], gparams['file']
	
	if kind == 'train':
		SIZE = int(gparams['file'].split('N')[0])
	else:
		SIZE = 1000
	FILE = f'{SIZE}N' + INPUT.split('N')[1]
	gparams['file'] = FILE

	if path != gparams['path']:
		index = path.index("training")
		def replace_line(file_name, text):
			os.chdir(path)
			lines = open(file_name, 'r').readlines()
			for i, _ in enumerate(lines):
				if 'path:' in _:
					line_num = i
					break
			lines[line_num] = 'path:' + text +'\n'
			out = open(file_name, 'w')
			out.writelines(lines)
			out.close()
			os.chdir(cwd)
		replace_line('parameters.txt', path[index:])
	PATH = gparams['path']
	KERNEL_SIZE = int(gparams['ks'])
	PADDING = (KERNEL_SIZE - 1)//2
	SHAPE = int(FILE.split('N')[1]) + 1
	BATCH_SIZE, D_in, Filters, D_out = SIZE, 1, int(gparams['filters']), SHAPE
	BLOCKS = int(gparams['blocks'])
	forcing = gparams['forcing']
	try:
		mean, std = gparams['mean'], gparams['std']
		norm = True
		transform_f = transforms.Normalize(mean, std)
		lg_dataset = get_data(gparams, kind=kind, transform_f=transform_f)
	except:
		norm = False
		lg_dataset = get_data(gparams, kind=kind)
	
	validateloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

	xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)

	# LOAD MODEL
	try:
		device = gparams['device']
	except:
		device = get_device()
	model = model(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS).to(device)
	model.load_state_dict(torch.load(PATH + '/model.pt'))
	model.eval()
	summary(model)

	MAE_a, MSE_a, MinfE_a, MAE_u, MSE_u, MinfE_u, pwe_a, pwe_u = [], [], [], [], [], [], [], []
	for batch_idx, sample_batch in enumerate(validateloader):
		f = sample_batch['f'].to(device)
		if norm == True:
			fn = sample_batch['fn'].to(device)
		else:
			fn = sample_batch['f'].to(device)
		u = sample_batch['u'].to(device)
		a = sample_batch['a'].to(device)
		a_pred = model(fn)
		u_pred = reconstruct(a_pred, phi)
		# a_pred = torch.zeros(BATCH_SIZE, D_in, D_out - 2).to(device)
		# u_pred = model(fn)
		# f_pred = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
		f_pred = torch.zeros(BATCH_SIZE, D_in, D_out).to(device)
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
		df2 = pd.DataFrame(gparams['losses'])
		df2.to_csv('losses.csv')
	except:
		pass

	sns.pairplot(df, corner=True, diag_kind="kde", kind="reg")
	plt.savefig('confusion_matrix.pdf', bbox_inches='tight', dpi=300)
	# plt.show()
	plt.close()

	rosetta = {
			   'MAE_a': 'MAE',
			   'MSE_a': 'Rel. $\\ell^{2}$',
			   'MinfE_a': '$\\ell^{\\infty}$',
			   'MAE_u': 'MAE',
			   'MSE_u': 'Rel. $\\ell^{2}$',
			   'MinfE_u': '$\\ell^{\\infty}$',
			  }

	columns = df.columns
	columns = columns[:-2]
	plt.figure(2, figsize=(14, 4))
	plt.suptitle("Error Histograms")
	for i, col in enumerate(columns[:-3]):
		if col in ('PWE_a', 'PWE_u'):
			continue
		plt.subplot(1, 3, i+1)
		sns.distplot(df[[f'{col}']], kde=False, color=blue)
		plt.grid(alpha=0.618)
		# plt.xlabel(f'{col}')
		plt.title(rosetta[f'{col}'])
		if i == 0:
			plt.ylabel('Count')
		else:
			plt.ylabel('')
		plt.xlim(0, df[f'{col}'].max())
		plt.xticks(rotation=90)
	plt.savefig('histogram_alphas.pdf', bbox_inches='tight', dpi=300)
	# plt.show()
	plt.close(2)

	plt.figure(3, figsize=(14, 4))
	plt.suptitle("Error Histograms")
	for i, col in enumerate(columns[-3:]):
		if col in ('PWE_a', 'PWE_u'):
			continue
		plt.subplot(1, 3, i+1)
		sns.distplot(df[[f'{col}']], kde=False, color=blue)
		plt.grid(alpha=0.618)
		# plt.xlabel(f'{col}')
		plt.title(rosetta[f'{col}'])
		if i == 0:
			plt.ylabel('Count')
		else:
			plt.ylabel('')
		plt.xlim(0, df[f'{col}'].max())
		plt.xticks(rotation=90)
	plt.savefig('histogram_solutions.pdf', bbox_inches='tight', dpi=300)
	# plt.show()
	plt.close(3)
	if gparams['model'] == 'ResNet' and gparams['blocks'] == 0:
		title = 'Linear'
	else:
		title = gparams['model']

	out_of_sample(EQUATION, SHAPE, a_pred, u_pred, f_pred, sample_batch, '.', title)
	
	try:
		loss_plot(gparams)
	except:
		print("Could not create loss plots.")
	os.chdir(cwd)
	return values