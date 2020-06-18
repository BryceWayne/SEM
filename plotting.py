#plotting.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sem.sem import legslbndm


def plotter(xx, sample, epoch, a=None, u=None, DE=None, title='alpha', ks=7, path='.'):
	def relative_l2(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)
	def relative_linf(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=np.inf)/np.linalg.norm(theoretical, ord=np.inf)
	def mae(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)
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
		plt.title(f'$\\alpha$ Example Epoch {epoch}\n'\
			      f'$\\alpha$ MAE Error: {np.round(mae_error_a, 6)}\n'\
			      f'$\\alpha$ Rel. $L_2$ Error: {np.round(float(l2_error_a), 6)}\n'\
			      f'$\\alpha$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_a), 6)}')
		plt.plot(x_, aa, 'ro-', label='$\\alpha$')
		plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		# plt.plot(xxx, ff, 'g-', label='$f$')
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
		# plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		# plt.plot(xxx, ff, 'g-', label='$f$')
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
		plt.title(f'$u$ Example Epoch {epoch}\n'\
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
		# plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		# plt.plot(xxx, ff, 'g-', label='$f$')
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
		plt.title(f'$f$ Example Epoch {epoch}\n'\
			      f'$f$ MAE Error: {np.round(mae_error_de, 6)}\n'\
			      f'$f$ Rel. $L_2$ Error: {np.round(float(l2_error_de), 6)}\n'\
			      f'$f$ Rel. $L_\\infty$ Error: {np.round(float(linf_error_de), 6)}')
		plt.plot(xxx, ff, 'ro-', label='$f$')
		plt.plot(xxx, de, 'bo', mfc='none', label='ODE')
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
		plt.plot(xxx, np.abs(ff-de), 'ro-', mfc='none', label='Error')
		# plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
		# plt.plot(xxx, ff, 'g-', label='$f$')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('Point-Wise Error')
		plt.legend(shadow=True)
		plt.savefig(f'{path}/pics/epoch{str(epoch).zfill(5)}_f_pwe.png', bbox_inches='tight')
		plt.close(3)


def loss_plot(losses, file, epoch, shape, ks, best_loss, title='alpha', path='.'):
	plt.figure(1, figsize=(10,6))
	x = list(range(1, len(losses)+1))
	plt.semilogy(x, losses, 'r-', label='Training')
	plt.xlabel('Epoch')
	plt.xlim(1, epoch)
	# plt.ylim(0, 1.168*max(losses))
	plt.grid(alpha=0.618)
	plt.ylabel('Loss')
	plt.legend(shadow=True)
	plt.title(f'Training Parameter {title}\nLoss vs. Epoch,$\\quad$Best Loss: {best_loss}\nFile: {file},$\\quad$Shape: {shape},$\\quad$Kernel: {ks}')
	plt.savefig(f'{path}/loss_{title}.png', bbox_inches='tight')
	# plt.show()
	plt.close(1)