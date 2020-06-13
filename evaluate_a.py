#evaluate.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import LG_1d
import argparse
import net.network as network
from net.data_loader import *
from sem.sem import *
from reconstruct import *
import subprocess
import pandas as pd



if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--file", type=str, default='1000N63')
parser.add_argument("--ks", type=int, default=7)
parser.add_argument("--input", type=str, default='20000N63')
parser.add_argument("--path", type=str, default='.')
parser.add_argument("--data", type=bool, default=False)
# parser.add_argument("--deriv", type=np.ndarray, default=np.zeros((1,1)))
args = parser.parse_args()

INPUT = args.input
FILE = args.file[:-2] + INPUT[-2:]
PATH = args.path
KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
SHAPE = int(FILE[-2:]) + 1
BATCH = int(args.file.split('N')[0])
N, D_in, Filters, D_out = BATCH, 1, 32, SHAPE
# LOAD MODEL
model = network.NetA(D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING).to(device)
print(PATH)
model.load_state_dict(torch.load(f'./{PATH}/{PATH}.pt'))
model.eval()

xx = legslbndm(D_out)
lepolys = gen_lepolys(D_out, xx)
lepoly_x = dx(D_out, xx, lepolys)
lepoly_xx = dxx(D_out, xx, lepolys)
phi = basis(SHAPE, lepolys)
phi_x = basis_x(SHAPE, phi, lepoly_x)
phi_xx = basis_xx(SHAPE, phi, lepoly_x)

def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)
def relative_linf(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=np.inf)/np.linalg.norm(theoretical, ord=np.inf)
def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)

# #Get out of sample data
if FILE.split('N')[1] != INPUT.split('N')[1]:
	FILE = '1000N' + INPUT.split('N')[1]
try:
	test_data = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
except:
	subprocess.call(f'python create_train_data.py --size {BATCH} --N {SHAPE - 1}', shell=True)
	test_data = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
testloader = torch.utils.data.DataLoader(test_data, batch_size=N, shuffle=False)

running_MAE_a, running_MAE_u, running_MSE_a, running_MSE_u, running_MinfE_a, running_MinfE_u = 0, 0, 0, 0, 0, 0
for batch_idx, sample_batch in enumerate(testloader):
	f = Variable(sample_batch['f']).to(device)
	u = Variable(sample_batch['u']).to(device)
	a = Variable(sample_batch['a']).to(device)
	a_pred = model(f)
	a = a.reshape(N, D_out-2)
	assert a_pred.shape == a.shape
	u_pred = reconstruct(N, a_pred, phi)
	u = u.reshape(N, D_out)
	assert u_pred.shape == u.shape
	DE = ODE2(1E-1, u_pred, a_pred, phi_x, phi_xx)
	f = f.reshape(N, D_out)
	assert DE.shape == f.shape
	a_pred = a_pred.to('cpu').detach().numpy()
	u_pred = u_pred.to('cpu').detach().numpy()
	a = a.to('cpu').detach().numpy()
	u = u.to('cpu').detach().numpy()
	for i in range(N):
		running_MAE_a += mae(a_pred[i,:], a[i,:])
		running_MSE_a += relative_l2(a_pred[i,:], a[i,:])
		running_MinfE_a += relative_linf(a_pred[i,:], a[i,:])
		running_MAE_u += mae(u_pred[i,:], u[i,:])
		running_MSE_u += relative_l2(u_pred[i,:], u[i,:])
		running_MinfE_u += relative_linf(u_pred[i,:], u[i,:])


print("***************************************************"\
	  f"\nAvg. alpha MAE: {np.round(running_MAE_a/N, 6)}\n"\
	  f"\nAvg. alpha MSE: {np.round(running_MSE_a/N, 6)}\n"\
	  f"\nAvg. alpha MinfE: {np.round(running_MinfE_a/N, 6)}\n"\
	  f"\nAvg. u MAE: {np.round(running_MAE_u/N, 6)}\n"\
	  f"\nAvg. u MSE: {np.round(running_MSE_u/N, 6)}\n"\
	  f"\nAvg. u MinfE: {np.round(running_MinfE_u/N, 6)}\n"\
	  "***************************************************")


xx = legslbndm(SHAPE-2)
ahat = a_pred[0,:]
ff = sample_batch['f'][0,0,:].to('cpu').detach().numpy()
aa = sample_batch['a'][0,0,:].to('cpu').detach().numpy()
mae_error_a = mae(ahat, aa)
l2_error_a = relative_l2(ahat, aa)
linf_error_a = relative_linf(ahat, aa)
plt.figure(1, figsize=(10,6))
plt.title(f'Example\nMAE Error: {np.round(float(mae_error_a), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
plt.plot(xx, aa, 'r-', label='$\\alpha$')
plt.plot(xx, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
xx_ = np.linspace(-1,1, len(xx)+2, endpoint=True)
# plt.plot(xx_, ff, 'g', label='$f$')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig(f'{PATH}/pics/a_ks{KERNEL_SIZE}_out_of_sample_alpha.png', bbox_inches='tight')
# plt.show()
plt.close()

uhat = u_pred[0,:]
uu = sample_batch['u'][0,0,:].to('cpu').detach().numpy()
mae_error_u = mae(uhat, uu)
l2_error_u = relative_l2(uhat, uu)
linf_error_u = relative_linf(uhat, uu)
xx = legslbndm(SHAPE)
plt.figure(2, figsize=(10,6))
plt.title(f'Example\nMAE Error: {np.round(float(mae_error_u), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_u), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_u), 6)}')
plt.plot(xx, uu, 'r-', label='$u$')
plt.plot(xx, uhat, 'bo', mfc='none', label='$\\hat{u}$')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig(f'{PATH}/pics/a_ks{KERNEL_SIZE}_out_of_sample_reconstruction.png', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure(3, figsize=(10,6))
de = DE[0,:].to('cpu').detach().numpy()
mae_error_de = mae(de, ff)
l2_error_de = relative_l2(de, ff)
linf_error_de = relative_linf(de, ff)
plt.title(f'Example\nMAE Error: {np.round(float(mae_error_de), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_de), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_de), 6)}')
xx_ = np.linspace(-1,1, len(ff), endpoint=True)
plt.plot(xx_, ff, 'g', label='$f$')
plt.plot(xx_, de, 'co', mfc='none', label='ODE')
plt.xlim(-1,1)
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(shadow=True)
plt.savefig(f'{PATH}/pics/a_ks{KERNEL_SIZE}_out_of_sample_DE.png', bbox_inches='tight')
# plt.show()
plt.close()

if args.data == True:
	COLS = ['TIMESTAMP', 'FOLDER', 'FILE', 'N', 'K.SIZE', 'BATCH', 'EPOCHS', 'AVG IT/S', 'LOSS', 'MAEa', 'MSEa', 'MIEa', 'MAEu', 'MSEu', 'MIEu']
	try:
		temp = pd.read_excel('temp.xlsx', dtype='object')
	except:
		temp = pd.DataFrame([], columns=COLS, dtype='object')
	d = {k:[i] for i, k in enumerate(COLS)}
	tempDF = pd.DataFrame.from_dict(d)
	temp = pd.concat([temp,tempDF])
	temp.at[temp.index[-1],'FOLDER'] = PATH[:-2]
	temp.at[temp.index[-1],'FILE'] = INPUT
	temp.at[temp.index[-1],'N'] = SHAPE
	temp.at[temp.index[-1],'K.SIZE'] = KERNEL_SIZE
	temp.at[temp.index[-1],'MAEa'] = running_MAE_a
	temp.at[temp.index[-1],'MSEa'] = running_MSE_a
	temp.at[temp.index[-1],'MIEa'] = running_MinfE_a
	temp.at[temp.index[-1],'MAEu'] = running_MAE_u
	temp.at[temp.index[-1],'MSEu'] = running_MSE_u
	temp.at[temp.index[-1],'MIEu'] = running_MinfE_u
	temp = temp[COLS]
	temp.to_excel('temp.xlsx')
	print('Done')