#plotting.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sem.sem import *


def relative_l2(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)
def relative_linf(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=np.inf)/np.linalg.norm(theoretical, ord=np.inf)
def mae(measured, theoretical):
	return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)

def color_scheme():
	# http://tableaufriction.blogspot.com/2012/11/finally-you-can-use-tableau-data-colors.html
	RED, BLUE, GREEN, PURPLE = '#ff2626', '#2626ff', '#26ff26', '#9326ff'
	return RED, BLUE, GREEN, PURPLE

def plotter(xx, sample, epoch, a=None, u=None, f=None, title='alpha', ks=5, path='.'):
	# https://www.colorhexa.com/
	# https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=4
	# http://vis.stanford.edu/papers/semantically-resonant-colors
	# https://medialab.github.io/iwanthue/
	RED, BLUE, GREEN, PURPLE = color_scheme()
	TEST  = {'color':RED, 'marker':'o', 'linestyle':'none', 'markersize': 3}
	VAL = {'color':BLUE, 'marker':'o', 'linestyle':'solid', 'mfc':'none'}
	aa = sample['a'][0,0,:].to('cpu').detach().numpy()
	uu = sample['u'][0,0,:].to('cpu').detach().numpy()
	ff = sample['f'][0,0,:].to('cpu').detach().numpy()
	x_ = legslbndm(len(xx)-2)
	xxx = np.linspace(-1,1, len(ff), endpoint=True)
	if a is not None:
		ahat = a[0,0,:].to('cpu').detach().numpy()
		mae_error_a = mae(ahat, aa)
		l2_error_a = relative_l2(ahat, aa)
		linf_error_a = relative_linf(ahat, aa)
		x_ = list(range(len(x_)))
		plt.figure(1, figsize=(10,6))
		plt.title(f'Model: {title}\n'\
				  f'$\\alpha$ Example Epoch {epoch}\n'\
			      f'$\\alpha$ MAE Error: {np.round(mae_error_a, 6)}\n'\
			      f'$\\alpha$ Rel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\n'\
			      f'$\\alpha$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
		plt.plot(x_, aa, **VAL, label='$\\alpha$')
		plt.plot(x_, ahat, **TEST, label='$\\hat{\\alpha}$')
		plt.xlim(x_[0], x_[-1])
		plt.grid(alpha=0.618)
		plt.xlabel('$i$')
		plt.ylabel('$\\alpha_i$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_a.png', bbox_inches='tight')
		plt.close(1)
		# plt.figure(1, figsize=(10,6))
		# plt.title(f'$\\alpha$ Example Epoch {epoch}\n'\
		# 	      f'$\\alpha$ Point-Wise Error: {np.round(np.sum(np.abs(aa-ahat))/len(x_), 6)}')
		# plt.plot(x_, np.abs(aa-ahat), 'ro-', mfc='none', label='Error')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_a_pwe.png', bbox_inches='tight')
		# plt.close(1)
	if u is not None:
		uhat = u[0,0,:].to('cpu').detach().numpy()
		mae_error_u = mae(uhat, uu)
		l2_error_u = relative_l2(uhat, uu)
		linf_error_u = relative_linf(uhat, uu)
		plt.figure(2, figsize=(10,6))
		plt.title(f'Model: {title}\n'\
				  f'$u$ Example Epoch {epoch}\n'\
			      f'$u$ MAE Error: {np.round(mae_error_u, 6)}\n'\
			      f'$u$ Rel. $L_2$ Error: {np.round(float(l2_error_u), 6)}\n'\
			      f'$u$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_u), 6)}')
		plt.plot(xx, uu, **VAL, label='$u$')
		plt.plot(xx, uhat.T, **TEST, label='$\\hat{u}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$u(x)$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_u.png', bbox_inches='tight')
		# plt.show()
		plt.close(2)
		# plt.figure(2, figsize=(10,6))
		# plt.title(f'$u$ Example Epoch {epoch}\n'\
		# 	      f'$u$ Point-Wise Error: {np.round(np.sum(np.abs(uu-uhat))/len(xx), 6)}')
		# plt.plot(xx, np.abs(uu-uhat), 'ro-', mfc='none', label='Error')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_u_pwe.png', bbox_inches='tight')
		# plt.close(2)
	if f is not None:
		f = f[0,0,:].to('cpu').detach().numpy()
		plt.figure(3, figsize=(10,6))
		mae_error_de = mae(f, ff)
		l2_error_de = relative_l2(f, ff)
		linf_error_de = relative_linf(f, ff)
		plt.title(f'Model: {title}\n'\
				  f'$f$ Example Epoch {epoch}\n'\
			      f'$f$ MAE Error: {np.round(mae_error_de, 6)}\n'\
			      f'$f$ Rel. $L_2$ Error: {np.round(float(l2_error_de), 6)}\n'\
			      f'$f$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_de), 6)}')
		plt.plot(xx[1:-1], ff[1:-1], **VAL, label='$f$')
		plt.plot(xx[1:-1], f[1:-1], **TEST, label='$\\hat{f}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$f(x)$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_f.png', bbox_inches='tight')
		# plt.show()
		plt.close(3)
		# plt.figure(3, figsize=(10,6))
		# plt.title(f'$f$ Example Epoch {epoch}\n'\
		# 	      f'$f$ Point-Wise Error: {np.round(np.sum(np.abs(ff-f))/len(xx), 6)}')
		# plt.plot(xx, np.abs(ff-f), 'ro-', mfc='none', label='Error')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_f_pwe.png', bbox_inches='tight')
		# plt.close(3)


def loss_plot(losses, file, epoch, shape, ks, best_loss, path, title='alpha'):
	RED, BLUE, GREEN, PURPLE = color_scheme()
	loss_a = losses['loss_a']
	loss_u = losses['loss_u']
	loss_f = losses['loss_f']
	loss_wf = losses['loss_wf']
	loss_train = losses['loss_train']
	loss_validate = losses['loss_validate']
	best_loss = np.round(float(best_loss), 9)

	N = int(file.split('N')[0])

	plt.figure(1, figsize=(10,6))
	x = list(range(1, len(loss_a)+1))
	plt.semilogy(x, np.array(loss_train), color=RED, label='Train')
	plt.semilogy(x, np.array(loss_validate), color=BLUE, label='Validate')
	plt.xlabel('Epoch')
	plt.xlim(1, epoch)
	plt.grid(alpha=0.618)
	plt.ylabel('Loss')
	plt.legend(shadow=True)
	plt.title(f'Loss vs. Epoch\nModel: {title}\nBest Loss: {best_loss}\nFile: {file},$\\quad$Shape: {shape},$\\quad$Kernel: {ks}')
	plt.savefig(f'{path}/loss train.png', bbox_inches='tight')
	# plt.show()
	plt.close(1)
	plt.figure(2, figsize=(10,6))
	x = list(range(1, len(loss_a)+1))
	if loss_a[-1] != 0:
		plt.semilogy(x, np.array(loss_a), color=RED, label='$\\hat{\\alpha}$')
	if loss_u[-1] != 0:
		plt.semilogy(x, np.array(loss_u), color=BLUE, label='$\\hat{u}$')
	if loss_f[-1] != 0:
		plt.semilogy(x, np.array(loss_f), color=GREEN, label='$\\hat{f}$')
	if loss_wf[-1] != 0:
		plt.semilogy(x, np.array(loss_wf), color=PURPLE, label='Weak Form')
	plt.xlabel('Epoch')
	plt.xlim(1, epoch)
	plt.grid(alpha=0.618)
	plt.ylabel('Loss')
	plt.legend(shadow=True)
	plt.title(f'Loss vs. Epoch\nModel: {title}\nFile: {file},$\\quad$Shape: {shape},$\\quad$Kernel: {ks}')
	plt.savefig(f'{path}/loss individual.png', bbox_inches='tight')
	# plt.show()
	plt.close(2)


def out_of_sample(equation, shape, a_pred, u_pred, f_pred, sample_batch, path, title='alpha'):
	RED, BLUE, GREEN, PURPLE = color_scheme()
	TEST  = {'color':RED, 'marker':'o', 'linestyle':'none', 'markersize': 3}
	VAL = {'color':BLUE, 'marker':'o', 'linestyle':'solid', 'mfc':'none'}
	PATH = path
	SHAPE = shape
	EQUATION = equation
	for picture in range(10):
		xx = legslbndm(SHAPE-2)
		ahat = a_pred[picture,0,:]
		ff = sample_batch['f'][picture,0,:].to('cpu').detach().numpy()
		aa = sample_batch['a'][picture,0,:].to('cpu').detach().numpy()
		mae_error_a = mae(ahat, aa)
		l2_error_a = relative_l2(ahat, aa)
		linf_error_a = relative_linf(ahat, aa)
		plt.figure(1, figsize=(10,6))
		plt.title(f'Out of Sample\nExample: {picture+1}, Model: {title}\nMAE Error: {np.round(float(mae_error_a), 6)}\n'\
				  f'Rel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\n'\
				  f'Rel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
		plt.plot(xx, aa, **VAL, label='$\\alpha$')
		plt.plot(xx, ahat, **TEST, label='$\\hat{\\alpha}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$u(x)$')
		plt.legend(shadow=True)
		plt.savefig(f'{PATH}/Out of Sample_0{picture}_a.png', bbox_inches='tight')
		plt.close()
		# plt.figure(1, figsize=(10,6))
		# plt.title(f'$\\alpha$ Point-Wise Error: {np.round(np.sum(np.abs(aa-ahat))/len(xx), 6)}')
		# plt.plot(xx, np.abs(aa-ahat), 'ro-', mfc='none', label='Error')
		# # plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		# # plt.plot(xxx, ff, 'g-', label='$f$')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{PATH}/{equation}_sample_a_pwe.png', bbox_inches='tight')
		# plt.close(1)


		uhat = u_pred[picture,0,:]
		uu = sample_batch['u'][picture,0,:].to('cpu').detach().numpy()
		mae_error_u = mae(uhat, uu)
		l2_error_u = relative_l2(uhat, uu)
		linf_error_u = relative_linf(uhat, uu)
		xx = legslbndm(SHAPE)
		plt.figure(2, figsize=(10,6))
		plt.title(f'Out of Sample\nExample: {picture+1}, Model: {title}\n'\
				  f'MAE Error: {np.round(float(mae_error_u), 6)}\n'\
				  f'Rel. $L_2$ Error: {np.round(float(l2_error_u), 6)}\n'\
				  f'Rel. $L_\\infty$ Error: {np.round(float(linf_error_u), 6)}')
		plt.plot(xx, uu, **VAL, label='$u$')
		plt.plot(xx, uhat, **TEST, label='$\\hat{u}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$u(x)$')
		plt.legend(shadow=True)
		plt.savefig(f'{PATH}/Out of Sample_0{picture}_u.png', bbox_inches='tight')
		plt.close()
		# plt.figure(2, figsize=(10,6))
		# plt.title(f'$u$ Point-Wise Error: {np.round(np.sum(np.abs(uu-uhat))/len(xx), 6)}')
		# plt.plot(xx, np.abs(uu-uhat), 'ro-', mfc='none', label='Error')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{PATH}/{equation}_sample_u_pwe.png', bbox_inches='tight')
		# plt.close(2)


		plt.figure(3, figsize=(10,6))
		f = f_pred[picture,0,:]
		mae_error_f = mae(f, ff)
		l2_error_f = relative_l2(f, ff)
		linf_error_f = relative_linf(f, ff)
		plt.title(f'Out of Sample\nExample: {picture+1}, Model: {title}\n'\
				  f'MAE Error: {np.round(float(mae_error_f), 6)}\n'\
				  f'Rel. $L_2$ Error: {np.round(float(l2_error_f), 6)}\n'\
				  f'Rel. $L_\\infty$ Error: {np.round(float(linf_error_f), 6)}')
		plt.plot(xx[1:-1], ff[1:-1], **VAL, label='$f$')
		plt.plot(xx[1:-1], f[1:-1], **TEST, label='$\\hat{f}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$u(x)$')
		plt.legend(shadow=True)
		plt.savefig(f'{PATH}/Out of Sample_0{picture}_f.png', bbox_inches='tight')
		plt.close()
		# plt.figure(3, figsize=(10,6))
		# plt.title(f'$f$ Point-Wise Error: {np.round(np.sum(np.abs(ff-f_pred))/len(xx_), 6)}')
		# plt.plot(xx, np.abs(ff-f_pred), 'ro-', mfc='none', label='Error')
		# plt.xlim(-1,1)
		# plt.grid(alpha=0.618)
		# plt.xlabel('$x$')
		# plt.ylabel('Point-Wise Error')
		# plt.legend(shadow=True)
		# plt.savefig(f'{PATH}/{equation}_sample_f_pwe.png', bbox_inches='tight')
		# plt.close(3)

