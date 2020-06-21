#logging.py
import pandas as pd
import subprocess
import numpy as np
from evaluate_a import *

def log_data(MODEL, KERNEL_SIZE, FILE, PATH, BLOCKS, EPSILON, FILTERS, DATA, EPOCHS, N, LOSS, AVG_ITER):
	# file_name, ks, path, filters, blocks, data
	model_metrics(MODEL, FILE, KERNEL_SIZE, PATH, EPSILON, FILTERS, BLOCKS, DATA)
	COLS = ['MODEL', 'TIMESTAMP', 'DATASET', 'FOLDER', 'SHAPE', 'BLOCKS', 'K.SIZE', 'FILTERS', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	if DATA == True:
		df = pd.read_excel('temp.xlsx')
		df.at[df.index[-1], 'MODEL'] = {MODEL}
		df.at[df.index[-1], 'AVG IT/S'] = {np.round(AVG_ITER, 1)}
		df.at[df.index[-1], 'LOSS'] = {np.round(LOSS, 6)}
		df.at[df.index[-1], 'EPOCHS'] = {EPOCHS}
		df.at[df.index[-1], 'BATCH'] = {N}
		df.at[df.index[-1], 'BLOCKS'] = {BLOCKS}
		df.at[df.index[-1], 'FILTERS'] = {FILTERS}
		df = df[COLS]
		_ = ['SHAPE', 'BLOCKS', 'K.SIZE', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
		for obj in _:
			df[obj] = df[obj].astype(float)
		df.to_excel('temp.xlsx')
	