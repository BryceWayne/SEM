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


def validate(gparams, model, optim, criterion, lepolys, phi, phi_x, phi_xx):
	device = gparams['device']
	VAL_SIZE = 1000
	SHAPE, EPSILON =  int(gparams['file'].split('N')[1]) + 1, gparams['epsilon']
	FILE, EQUATION = f'{VAL_SIZE}N{SHAPE-1}', gparams['equation']
	BATCH_SIZE, D_in, Filters, D_out = VAL_SIZE, 1, gparams['filters'], SHAPE
	NBFUNCS, SD = gparams['nbfuncs'], gparams['sd']
	A, F, U, WF = gparams['A'], gparams['F'], gparams['U'], gparams['WF']
	criterion_a, criterion_u = criterion['a'], criterion['u']
	criterion_f, criterion_wf = criterion['f'], criterion['wf']
	test_data = get_data(gparams, kind='validate')
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
				f_pred = ODE2(EPSILON, u_pred, a_pred, phi_x, phi_xx, equation=EQUATION)
				loss_f = F*criterion_f(f_pred, f)
			else:
				f_pred, loss_f = None, 0
			if WF != 0 and EQUATION != 'BurgersT':
				LHS, RHS = weak_form2(EPSILON, SHAPE, f, u_pred, a_pred, lepolys, phi, phi_x, equation=EQUATION, nbfuncs=NBFUNCS)
				loss_wf = WF*criterion_wf(LHS, RHS)
			else:
				loss_wf = 0
			loss = loss_a + loss_u + loss_f + loss_wf
			return np.round(float(loss.to('cpu').detach()), 8)
		loss += closure(f, a, u)
	optim.zero_grad()
	return loss


def model_stats(path, kind='train'):
	red, blue, green, purple = color_scheme()
	TEST  = {'color':red, 'marker':'o', 'linestyle':'none', 'markersize': 3}
	VAL = {'color':blue, 'marker':'o', 'linestyle':'solid', 'mfc':'none'}
	# device = get_device()
	cwd = os.getcwd()
	os.chdir(path)
	with open("parameters.txt", 'r') as f:
		text = f.readlines()
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
	# pprint(gparams)	
	if gparams['model'] == 'ResNet':
		model = ResNet
	elif gparams['model'] == 'NetA':
		model = NetA
	elif gparams['model'] == 'NetB':
		model = NetB
	elif gparams['model'] == 'NetC':
		model = NetC
	
	EQUATION, EPSILON, INPUT = gparams['equation'], gparams['epsilon'], gparams['file']
	
	if kind == 'train':
		SIZE = int(gparams['file'].split('N')[0])
	else:
		SIZE = 1000
	FILE = f'{SIZE}N' + INPUT.split('N')[1]
	gparams['file'] = FILE
	PATH = gparams['path']
	KERNEL_SIZE = int(gparams['ks'])
	PADDING = (KERNEL_SIZE - 1)//2
	SHAPE = int(FILE.split('N')[1]) + 1
	BATCH_SIZE, D_in, Filters, D_out = SIZE, 1, int(gparams['filters']), SHAPE
	BLOCKS = int(gparams['blocks'])

	xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx = basis_vectors(D_out, equation=EQUATION)
	# LOAD MODEL
	device = gparams['device']
	model = model(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS).to(device)
	model.load_state_dict(torch.load(PATH + '/model.pt'))
	model.eval()

	test_data = get_data(gparams, kind=kind)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

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
			# if relative_l2(u_pred[i,0,:], u[i,0,:]) > 0.135:
			# 	plt.figure(figsize=(10,6))
			# 	plt.title(f'MAE: {np.round(MAE_u[-1], 6)}, '\
			# 			  f'Rel. MSE: {np.round(float(MSE_u[-1]), 6)}, '\
			# 			  f'L$_\\infty$E: {np.round(float(MinfE_u[-1]), 6)}')
			# 	plt.plot(xx, u[i,0,:], **VAL, label='True')
			# 	plt.plot(xx, u_pred[i,0,:], **TEST, label='Pred')
			# 	plt.legend()
			# 	plt.xlim(-1, 1)
			# 	plt.show()
	
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
	import matplotlib
	matplotlib.rcParams['savefig.dpi'] = 300
	import seaborn as sns
	sns.pairplot(df, corner=True, diag_kind="kde", kind="reg")
	plt.savefig('confusion_matrix.png', bbox_inches='tight')
	# plt.show()
	plt.close()

	rosetta = {
			   'MAE_a': 'MAE',
			   'MSE_a': 'Rel. $L_{2}$',
			   'MinfE_a': '$L_{\\infty}$',
			   'MAE_u': 'MAE',
			   'MSE_u': 'Rel. $L_{2}$',
			   'MinfE_u': '$L_{\\infty}$',
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
	plt.savefig('histogram_alphas.png', bbox_inches='tight')
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
	plt.savefig('histogram_solutions.png', bbox_inches='tight')
	# plt.show()
	plt.close(3)
	if gparams['model'] == 'ResNet' and gparams['blocks'] == 0:
		title = 'Linear'
	else:
		title = gparams['model']
	out_of_sample(EQUATION, SHAPE, a_pred, u_pred, f_pred, sample_batch, '.', title)
	os.chdir(cwd)
	return values