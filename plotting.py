#plotting.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sem.sem import legslbndm


def plotter(xx, sample, a_pred, u_pred, epoch, DE=None):
	def relative_l2(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=2)/np.linalg.norm(theoretical, ord=2)
	def mae(measured, theoretical):
		return np.linalg.norm(measured-theoretical, ord=1)/len(theoretical)
	ahat = a_pred[0,:].to('cpu').detach().numpy()
	aa = sample['a'][0,0,:].to('cpu').detach().numpy()
	uu = sample['u'][0,0,:].to('cpu').detach().numpy()
	ff = sample['f'][0,0,:].to('cpu').detach().numpy()[1:31]
	x_ = legslbndm(len(xx)-2)
	xxx = np.linspace(-1,1, len(ff), endpoint=True)
	mae_error_a = mae(ahat, aa)
	l2_error_a = relative_l2(ahat, aa)
	plt.figure(1, figsize=(10,6))
	plt.title(f'Alphas Example Epoch {epoch}\n'\
		      f'Alphas MAE Error: {np.round(mae_error_a, 6)}\n'\
		      f'Alphas Rel. $L_2$ Error: {np.round(float(l2_error_a), 6)}')
	plt.plot(x_, aa, 'r-', mfc='none', label='$\\alpha$')
	plt.plot(x_, ahat, 'bo', mfc='none', label='$\\hat{\\alpha}$')
	# plt.plot(xxx, ff, 'g-', label='$f$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'./pics/alphas_epoch{epoch}.png')
	plt.close(1)
	uhat = u_pred[0,:].to('cpu').detach().numpy()
	mae_error_u = mae(uhat, uu)
	l2_error_u = relative_l2(uhat, uu)
	# xx = legslbndm(len(xx))
	plt.figure(2, figsize=(10,6))
	plt.title(f'Reconstruction Example Epoch {epoch}\n'\
		      f'Reconstruction MAE Error: {np.round(mae_error_u, 6)}\n'\
		      f'Reconstruction Rel. $L_2$ Error: {np.round(float(l2_error_u), 6)}')
	plt.plot(xx, uu, 'r-', mfc='none', label='$u$')
	plt.plot(xx, uhat.T, 'bo', mfc='none', label='$\\hat{u}$')
	plt.xlim(-1,1)
	plt.grid(alpha=0.618)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.savefig(f'./pics/reconstruction_epoch{epoch}.png')
	# plt.show()
	plt.close(2)

	if DE is not None:
		de = DE[0,:].to('cpu').detach().numpy()
		plt.figure(3, figsize=(10,6))
		mae_error_de = mae(de, ff)
		l2_error_de = relative_l2(de, ff)
		plt.title(f'DE Example Epoch {epoch}\n'\
			      f'DE MAE Error: {np.round(mae_error_de, 6)}\n'\
			      f'DE Rel. $L_2$ Error: {np.round(float(l2_error_de), 6)}')
		plt.plot(xxx, ff, 'g-', label='$f$')
		plt.plot(xxx, de, 'co', mfc='none', label='ODE')
		plt.xlim(-1,1)
		plt.grid(alpha=0.618)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.legend(shadow=True)
		plt.savefig(f'./pics/DE_epoch{epoch}.png')
		# plt.show()
		plt.close(3)