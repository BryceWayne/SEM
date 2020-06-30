#data_logging.py
import pandas as pd
import subprocess
import numpy as np
from evaluate_a import *

def log_data(EQUATION, MODEL, KERNEL_SIZE, FILE, PATH, BLOCKS, EPSILON, FILTERS, EPOCHS, N, LOSS, AVG_ITER, LOSSES, LOSS_TYPE):
	data = model_metrics(EQUATION, MODEL, FILE, KERNEL_SIZE, PATH, EPSILON, FILTERS, BLOCKS)
	if MODEL == ResNet:
		data['MODEL'] = 'ResNet'
	elif MODEL == NetA:
		data['MODEL'] = 'NetA'
	data['AVG IT/S'] = np.round(AVG_ITER, 1)
	data['LOSS'] = np.round(LOSS, 6)
	data['LOSS_TYPE'] = LOSS_TYPE
	data['EPOCHS'] = EPOCHS
	data['BATCH'] = N
	data['BLOCKS'] = BLOCKS
	data['FILTERS'] = FILTERS

	COLS = ['EQUATION', 'MODEL', 'LOSS TYPE', 'TIMESTAMP', 'DATASET', 'FOLDER', 'SHAPE', 'BLOCKS', 'K.SIZE', 'FILTERS', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
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
	df.to_excel('temp.xlsx')

	df = pd.DataFrame(LOSSES)
	df.to_excel(PATH + '/losses.xlsx')
	