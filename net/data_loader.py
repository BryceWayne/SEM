#data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pprint import pprint


class LGDataset():
    """Legendre-Galerkin Dataset."""

    def __init__(self, pickle_file, shape=64, transform_f=None, transform_a=None, subsample=None):
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        with open('./data/' + pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
            self.data = self.data[:,:]
        self.transform_f = transform_f
        self.transform_a = transform_a
        self.subsample = subsample
        self.shape = shape
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        u = torch.Tensor([self.data[:,0][idx]]).reshape(1, self.shape)
        f = torch.Tensor([self.data[:,1][idx]]).reshape(1, self.shape)
        a = torch.Tensor([self.data[:,2][idx]]).reshape(1, self.shape-2)
        p = torch.Tensor([self.data[:,3][idx]]).reshape(1, 4)
        # if self.subsample:
        #     a = a[:,:self.subsample]
        if self.transform_f:
            f = f.view(1, 1, self.shape)
            f = self.transform_f(f).view(1, self.shape)
        # if self.transform_a:
        #     a = a.view(1, 1, 64)
        #     a = self.transform_a(a).view(1, 64)
        sample = {'u': u, 'f': f, 'a': a, 'p': p}
        return sample


def normalize(pickle_file, dim):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    if dim == 'f':
        dim = 1
    elif dim == 'a':
        dim = 2
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)
    data = load_obj(pickle_file)
    f = torch.Tensor(data[:,dim,:])
    sum_ = torch.sum(f, dim=[0])
    sum_of_square = torch.sum(f ** 2, dim=[0])
    fst_moment = (f.shape[0] * fst_moment + sum_) / (f.shape[0] + f.shape[1])
    snd_moment = (f.shape[0] * snd_moment + sum_of_square) / (f.shape[0] + f.shape[1])
    return fst_moment.mean().item(), torch.sqrt(snd_moment - fst_moment ** 2).mean().item()
def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[:,:,0,:]
    return data
def show_solution(solution):
	x, y = solution[0], solution[1]
	plt.figure(1, figsize=(10,6))
	plt.title('Exact Solution')
	plt.plot(x, y, label='Exact')
	plt.xlabel('$x$')
	plt.xlim(x.min(), x.max())
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.grid(alpha=0.618)
	plt.title('Exact Solution')
	plt.show()

def debug():
	lg_dataset = LGDataset(pickle_file='1000')
	for i in range(len(lg_dataset)):
	    sample = lg_dataset[i]
	    show_solution([sample['x'], sample['u']])
	    if i == 10:
	        break