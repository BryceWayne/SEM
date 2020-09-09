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

# def log_data(gparams, model):
# 	equation, kernel_size, path, file = gparams['equation'], gparams['ks'], gparams['path'], gparams['file']
# 	epsilon, filters, blocks, sd = gparams['epsilon'], gparams['filters'], gparams['blocks'], gparams['sd']
# 	epochs, npfuncs, nparams = gparams['epochs'], gparams['nbfuncs'], gparams['nparams']
# 	batch_size, loss, avgIter = gparams['batch_size'], gparams['loss'], gparams['avgIter']
# 	losses, loss_type = gparams['losses'], gparams['loss_type']
# 	data = model_metrics(gparams, model)
# 	data['AVG IT/S'] = np.round(avgIter, 1)
# 	data['LOSS'] = np.round(loss, 6)
# 	data['LOSS_TYPE'] = loss_type
# 	data['EPOCHS'] = epochs
# 	data['BATCH'] = batch_size
# 	data['BLOCKS'] = blocks
# 	data['FILTERS'] = filters
# 	data['EPSILON'] = epsilon
# 	data['NBFUNCS'] = nbfuncs
# 	data['NPARAMS'] = nparams
# 	return df


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
