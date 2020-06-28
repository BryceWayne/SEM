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


def plotter(xx, sample, epoch, a=None, u=None, DE=None, title='alpha', ks=7, path='.'):
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
		plt.figure(1, figsize=(10,6))
		plt.title(f'Model: {title}\n'\
				  f'$\\alpha$ Example Epoch {epoch}\n'\
			      f'$\\alpha$ MAE Error: {np.round(mae_error_a, 6)}\n'\
			      f'$\\alpha$ Rel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\n'\
			      f'$\\alpha$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
		plt.plot(x_, aa, 'ro-', label='$\\alpha$')
		plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_a.png', bbox_inches='tight')
		plt.close(1)
		plt.figure(1, figsize=(10,6))
		plt.title(f'$\\alpha$ Example Epoch {epoch}\n'\
			      f'$\\alpha$ Point-Wise Error: {np.round(np.sum(np.abs(aa-ahat))/len(x_), 6)}')
		plt.plot(x_, np.abs(aa-ahat), 'ro-', mfc='none', label='Error')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('Point-Wise Error')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_a_pwe.png', bbox_inches='tight')
		plt.close(1)
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
		plt.plot(xx, uu, 'ro-', label='$u$')
		plt.plot(xx, uhat.T, 'bo', mfc='none', label='$\\hat{u}$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_u.png', bbox_inches='tight')
		# plt.show()
		plt.close(2)
		plt.figure(2, figsize=(10,6))
		plt.title(f'$u$ Example Epoch {epoch}\n'\
			      f'$u$ Point-Wise Error: {np.round(np.sum(np.abs(uu-uhat))/len(xx), 6)}')
		plt.plot(xx, np.abs(uu-uhat), 'ro-', mfc='none', label='Error')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('Point-Wise Error')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_u_pwe.png', bbox_inches='tight')
		plt.close(2)
	if DE is not None:
		de = DE[0,0,:].to('cpu').detach().numpy()
		plt.figure(3, figsize=(10,6))
		mae_error_de = mae(de, ff)
		l2_error_de = relative_l2(de, ff)
		linf_error_de = relative_linf(de, ff)
		plt.title(f'Model: {title}\n'\
				  f'$f$ Example Epoch {epoch}\n'\
			      f'$f$ MAE Error: {np.round(mae_error_de, 6)}\n'\
			      f'$f$ Rel. $L_2$ Error: {np.round(float(l2_error_de), 6)}\n'\
			      f'$f$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_de), 6)}')
		plt.plot(xx, ff, 'ro-', label='$f$')
		plt.plot(xx, de, 'bo', mfc='none', label='ODE')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_f.png', bbox_inches='tight')
		# plt.show()
		plt.close(3)
		plt.figure(3, figsize=(10,6))
		plt.title(f'$f$ Example Epoch {epoch}\n'\
			      f'$f$ Point-Wise Error: {np.round(np.sum(np.abs(ff-de))/len(xxx), 6)}')
		plt.plot(xx, np.abs(ff-de), 'ro-', mfc='none', label='Error')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('Point-Wise Error')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_f_pwe.png', bbox_inches='tight')
		plt.close(3)


def loss_plot(losses, file, epoch, shape, ks, best_loss, path):
	loss_a = losses['loss_a']
	loss_u = losses['loss_u']
	loss_f = losses['loss_f']
	loss_wf = losses['loss_wf']
	loss_train = losses['loss_train']
	loss_validate = losses['loss_validate']
	N = int(file.split('N')[0])

	plt.figure(1, figsize=(10,6))
	x = list(range(1, len(loss_a)+1))
	plt.semilogy(x, np.array(loss_train), label='Train')
	plt.semilogy(x, np.array(loss_validate), label='Validate')
	plt.xlabel('Epoch')
	plt.xlim(1, epoch)
	plt.grid(alpha=0.618)
	plt.ylabel('Loss')
	plt.legend(shadow=True)
	plt.title(f'Loss vs. Epoch,$\\quad$Best Loss: {best_loss}\nFile: {file},$\\quad$Shape: {shape},$\\quad$Kernel: {ks}')
	plt.savefig(f'{path}/loss.png', bbox_inches='tight')
	# plt.show()
	plt.close(1)
	plt.figure(2, figsize=(10,6))
	x = list(range(1, len(loss_a)+1))
	plt.semilogy(x, np.array(loss_a), label='$\\hat{\\alpha}$')
	plt.semilogy(x, np.array(loss_u), label='$\\hat{u}$')
	# plt.semilogy(x, loss_f, label='$\\hat{f}$')
	plt.semilogy(x, np.array(loss_wf), label='Weak Form')
	plt.xlabel('Epoch')
	plt.xlim(1, epoch)
	plt.grid(alpha=0.618)
	plt.ylabel('Loss')
	plt.legend(shadow=True)
	plt.title(f'Loss vs. Epoch\nFile: {file},$\\quad$Shape: {shape},$\\quad$Kernel: {ks}')
	plt.savefig(f'{path}/loss_individual.png', bbox_inches='tight')
	# plt.show()
	plt.close(2)


def out_of_sample(equation, shape, a_pred, u_pred, f_pred, sample_batch, path):
	PATH = path
	SHAPE = shape
	EQUATION = equation
	xx = legslbndm(SHAPE-2)
	ahat = a_pred[0,0,:]
	ff = sample_batch['f'][0,0,:].to('cpu').detach().numpy()
	aa = sample_batch['a'][0,0,:].to('cpu').detach().numpy()
	mae_error_a = mae(ahat, aa)
	l2_error_a = relative_l2(ahat, aa)
	linf_error_a = relative_linf(ahat, aa)
	plt.figure(1, figsize=(10,6))
	plt.title(f'Out of Sample Example\nMAE Error: {np.round(float(mae_error_a), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
	plt.plot(xx, aa, 'ro-', label='$\\alpha$')
	plt.plot(xx, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
	xx_ = np.linspace(-1,1, len(xx)+2, endpoint=True)
	# plt.plot(xx_, ff, 'g', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_a.png', bbox_inches='tight')
	# plt.show()
	plt.close()
	plt.figure(1, figsize=(10,6))
	plt.title(f'$\\alpha$ Point-Wise Error: {np.round(np.sum(np.abs(aa-ahat))/len(xx), 6)}')
	plt.plot(xx, np.abs(aa-ahat), 'ro-', mfc='none', label='Error')
	# plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
	# plt.plot(xxx, ff, 'g-', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('Point-Wise Error')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_a_pwe.png', bbox_inches='tight')
	plt.close(1)


	uhat = u_pred[0,0,:]
	uu = sample_batch['u'][0,0,:].to('cpu').detach().numpy()
	mae_error_u = mae(uhat, uu)
	l2_error_u = relative_l2(uhat, uu)
	linf_error_u = relative_linf(uhat, uu)
	xx = legslbndm(SHAPE)
	plt.figure(2, figsize=(10,6))
	plt.title(f'Out of Sample Example\nMAE Error: {np.round(float(mae_error_u), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_u), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_u), 6)}')
	plt.plot(xx, uu, 'ro-', label='$u$')
	plt.plot(xx, uhat, 'bo', mfc='none', label='$\\hat{u}$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_u.png', bbox_inches='tight')
	# plt.show()
	plt.close()
	plt.figure(2, figsize=(10,6))
	plt.title(f'$u$ Point-Wise Error: {np.round(np.sum(np.abs(uu-uhat))/len(xx), 6)}')
	plt.plot(xx, np.abs(uu-uhat), 'ro-', mfc='none', label='Error')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('Point-Wise Error')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_u_pwe.png', bbox_inches='tight')
	plt.close(2)


	plt.figure(3, figsize=(10,6))
	f_pred = f_pred[0,0,:]
	mae_error_f = mae(f_pred, ff)
	l2_error_f = relative_l2(f_pred, ff)
	linf_error_f = relative_linf(f_pred, ff)
	plt.title(f'Out of Sample Example\nMAE Error: {np.round(float(mae_error_f), 6)}\nRel. $L_2$ Error: {np.round(float(l2_error_f), 6)}\nRel. $L_\\infty$ Error: {np.round(float(linf_error_f), 6)}')
	plt.plot(xx, ff, 'ro-', label='$f$')
	plt.plot(xx, f_pred, 'bo', mfc='none', label='ODE')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_f.png', bbox_inches='tight')
	# plt.show()
	plt.close()
	plt.figure(3, figsize=(10,6))
	plt.title(f'$f$ Point-Wise Error: {np.round(np.sum(np.abs(ff-f_pred))/len(xx_), 6)}')
	plt.plot(xx, np.abs(ff-f_pred), 'ro-', mfc='none', label='Error')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('Point-Wise Error')
	plt.legend(shadow=True)
	plt.savefig(f'{PATH}/{equation}_sample_f_pwe.png', bbox_inches='tight')
	plt.close(3)

