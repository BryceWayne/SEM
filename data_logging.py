#logging.py
import pandas as pd
import subprocess


def log_data(KERNEL_SIZE, FILE, PATH, BLOCKS, FILTERS, DATA, EPOCHS, N, LOSS, AVG_ITER):
	COLS = ['TIMESTAMP', 'DATASET', 'FOLDER', 'SHAPE', 'BLOCKS', 'K.SIZE', 'FILTERS', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	if DATA == True:
		subprocess.call(f'python evaluate_a.py --ks {KERNEL_SIZE} --input {FILE} --path {PATH} --blocks {BLOCKS} --filters {FILTERS} --data True', shell=True)
		df = pd.read_excel('temp.xlsx')
		df.at[df.index[-1],'AVG IT/S'] = {AVG_ITER}
		df.at[df.index[-1],'LOSS'] = {LOSS}
		df.at[df.index[-1],'EPOCHS'] = {EPOCHS}
		df.at[df.index[-1],'BATCH'] = {N}
		df.at[df.index[-1], 'BLOCKS'] = {BLOCKS}
		df.at[df.index[-1], 'FILTERS'] = {FILTERS}
		df = df[COLS]
		_ = ['SHAPE', 'BLOCKS', 'K.SIZE', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
		for obj in _:
			df[obj] = df[obj].astype(float)
		df.to_excel('temp.xlsx')
	else:
		subprocess.call(f'python evaluate_a.py --ks {KERNEL_SIZE} --input {FILE} --path {PATH} --blocks {BLOCKS}  --filters {FILTERS}', shell=True)