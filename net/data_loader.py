#data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pprint import pprint
import subprocess


def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[:,:,0,:]
    return data


def get_data(gparams, kind='train'):
    equation, file, sd = gparams['equation'], gparams['file'], gparams['sd']
    shape, epsilon = int(file.split('N')[1]) + 1, gparams['epsilon']
    if kind == 'validate':
        size = 1000
        file = f'{size}N{shape-1}'
    else:
        size = int(file.split('N')[0])
    try:
        data = LGDataset(equation=equation, pickle_file=file, shape=shape, kind=kind, sd=sd)
    except:
        subprocess.call(f'python create_train_data.py --equation {equation} --size {size}'\
                        f' --N {shape - 1} --eps {epsilon} --kind {kind} --sd {sd}', shell=True)
        data = LGDataset(equation=equation, pickle_file=file, shape=shape, kind=kind, sd=sd)
    return data


class LGDataset():
    """Legendre-Galerkin Dataset."""
    def __init__(self, equation, pickle_file, shape=64, transform_f=None, transform_a=None, kind='train', sd=1):
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        if equation == 'Burgers':
            pickle_file += f'sd{sd}'
        with open(f'./data/{equation}/{kind}/' + pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
            self.data = self.data[:,:]
        self.transform_f = transform_f
        self.transform_a = transform_a
        self.shape = shape
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        L = len(self.data[:,3][idx])
        if torch.is_tensor(idx):
            idx = idx.tolist()
        u = torch.Tensor([self.data[:,0][idx]]).reshape(1, self.shape)
        f = torch.Tensor([self.data[:,1][idx]]).reshape(1, self.shape)
        a = torch.Tensor([self.data[:,2][idx]]).reshape(1, self.shape-2)
        p = torch.Tensor([self.data[:,3][idx]]).reshape(1, L)
        if self.transform_f:
            f = f.view(1, 1, self.shape)
            f = self.transform_f(f).view(1, self.shape)
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
    with open(f'./data/{equation}/{kind}/' + pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
            self.data = self.data[:,:]
    f = torch.Tensor(data[:,dim])
    sum_ = torch.sum(f, dim=[0])
    sum_of_square = torch.sum(f ** 2, dim=[0])
    fst_moment = (f.shape[0] * fst_moment + sum_) / (f.shape[0] + f.shape[1])
    snd_moment = (f.shape[0] * snd_moment + sum_of_square) / (f.shape[0] + f.shape[1])
    return fst_moment.mean().item(), torch.sqrt(snd_moment - fst_moment ** 2).mean().item()

