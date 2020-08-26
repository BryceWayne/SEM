#data_logging.py
import pandas as pd
import pickle
import subprocess
import numpy as np
from evaluate import *
import os, json


def record_path(path):
	entry = str(path) + '\n'
	with open("paths.txt", 'a') as f:
		f.write(entry)


def log_loss(losses, loss_a, loss_u, loss_f, loss_wf, loss_train, loss_validate, dataset):
	if type(loss_a) == int:
		losses['loss_a'].append(loss_a/dataset)
	else:
		losses['loss_a'].append(loss_a.item()/dataset)
	if type(loss_u) == int:
		losses['loss_u'].append(loss_u/dataset)
	else:
		losses['loss_u'].append(loss_u.item()/dataset)
	if type(loss_f) == int:
		losses['loss_f'].append(loss_f/dataset) 
	else:
		losses['loss_f'].append(loss_f.item()/dataset) 
	if type(loss_wf) == int:
		losses['loss_wf'].append(loss_wf/dataset) 
	else:
		losses['loss_wf'].append(loss_wf.item()/dataset)
	losses['loss_train'].append(loss_train.item()/dataset)
	losses['loss_validate'].append(loss_validate.item()/1000)
	return losses

def log_data(gparams, model):
	equation, kernel_size, path, file = gparams['equation'], gparams['ks'], gparams['path'], gparams['file']
	epsilon, filters, blocks, sd = gparams['epsilon'], gparams['filters'], gparams['blocks'], gparams['sd']
	epochs, npfuncs, nparams = gparams['epochs'], gparams['nbfuncs'], gparams['nparams']
	batch_size, loss, avgIter = gparams['batch_size'], gparams['loss'], gparams['avgIter']
	losses, loss_type = gparams['losses'], gparams['loss_type']
	data = model_metrics(gparams, model)
	data['AVG IT/S'] = np.round(avgIter, 1)
	data['LOSS'] = np.round(loss, 6)
	data['LOSS_TYPE'] = loss_type
	data['EPOCHS'] = epochs
	data['BATCH'] = batch_size
	data['BLOCKS'] = blocks
	data['FILTERS'] = filters
	data['EPSILON'] = epsilon
	data['NBFUNCS'] = nbfuncs
	data['NPARAMS'] = nparams

	return df

def loss_log(params, losses, df):
	try:
		with open('./losses.pkl', 'rb') as f:
			data = pickle.load(f)
	except:
		data = {}

	entry = {'losses': losses}
	entry['EQUATION'] = params['EQUATION']
	if params['MODEL'] == NetA:
		entry['MODEL'] = 'NetA'
	elif params['MODEL'] == NetB:
		entry['MODEL'] = 'NetB'
	elif params['MODEL'] == NetC:
		entry['MODEL'] = 'NetC'
	elif params['MODEL'] == ResNet:
		entry['MODEL'] = 'ResNet'
		
	entry['KERNEL_SIZE'] = params['KERNEL_SIZE']
	entry['BLOCKS'] = params['BLOCKS']
	entry['FILTERS'] = params['FILTERS']
	entry['EPSILON'] = params['EPSILON']
	entry['EPOCHS'] = params['EPOCHS']
	entry['LOSS_TYPE'] = params['LOSS_TYPE']
	entry['NBFUNCS'] = params['NBFUNCS']
	entry['NPARAMS'] = params['NPARAMS']
	for _ in ['MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']:
		val = df[_].tolist()
		entry[_] = val[-1]
	data[params['PATH']] = entry

	with open(f'./losses.pkl', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def log_gparams(gparams):
	cwd = os.getcwd()
	os.chdir(gparams['path'])
	with open('parameters.txt', 'w') as f:
		for k, v in gparams.items():
			if k == 'losses':
				df = pd.DataFrame(gparams['losses'])
				df.to_csv('losses.csv')
			else:
				entry = f"{k}:{v}\n"
				f.write(entry)
	os.chdir(cwd)
