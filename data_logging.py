#data_logging.py
import pandas as pd
import pickle
import subprocess
import numpy as np
from evaluate import *


def global_parameters():
	return None

	
def log_data(EQUATION, MODEL, KERNEL_SIZE, FILE, PATH, BLOCKS, EPSILON, FILTERS, EPOCHS, N, LOSS, AVG_ITER, LOSSES, LOSS_TYPE):
	data = model_metrics(EQUATION, MODEL, FILE, KERNEL_SIZE, PATH, EPSILON, FILTERS, BLOCKS)
	data['AVG IT/S'] = np.round(AVG_ITER, 1)
	data['LOSS'] = np.round(LOSS, 6)
	data['LOSS_TYPE'] = LOSS_TYPE
	data['EPOCHS'] = EPOCHS
	data['BATCH'] = N
	data['BLOCKS'] = BLOCKS
	data['FILTERS'] = FILTERS
	data['EPSILON'] = EPSILON

	COLS = ['EQUATION', 'MODEL', 'LOSS_TYPE', 'TIMESTAMP', 'DATASET', 'FOLDER', 'SHAPE', 'BLOCKS', 'K.SIZE', 'FILTERS', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	try:
		df = pd.read_excel('temp.xlsx', ignore_index=True)
	except:
		df = pd.DataFrame([], columns=COLS)			
	entries = df.to_dict('records')
	entries.append(data)

	df = pd.DataFrame(entries)
	df = df[COLS]
	_ = ['SHAPE', 'BLOCKS', 'K.SIZE', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	for obj in _:
		df[obj] = df[obj].astype(float)
	df.to_excel('log_data.xlsx')
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
	elif params['MODEL'] == ResNet:
		entry['MODEL'] = 'ResNet'
	entry['KERNEL_SIZE'] = params['KERNEL_SIZE']
	entry['BLOCKS'] = params['BLOCKS']
	entry['FILTERS'] = params['FILTERS']
	entry['EPSILON'] = params['EPSILON']
	entry['LOSS_TYPE'] = params['LOSS_TYPE']
	for _ in ['MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']:
		val = df[_].tolist()
		entry[_] = val[-1]
	data[params['PATH']] = entry

	with open(f'./losses.pkl', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
