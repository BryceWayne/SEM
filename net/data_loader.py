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


def get_data(gparams, kind='train', transform_f=None):
    equation, file, sd = gparams['equation'], gparams['file'], gparams['sd']
    if sd == 1:
        sd = 1.0
    shape, epsilon = int(file.split('N')[1]) + 1, gparams['epsilon']
    forcing = gparams['forcing']
    if kind == 'validate':
        size = 1000
        file = f'{size}N{shape-1}'
    else:
        size = int(file.split('N')[0])
    try:
        data = LGDataset(equation=equation, pickle_file=file, shape=shape, kind=kind, sd=sd, forcing=forcing, transform_f=transform_f)
    except:
        subprocess.call(f'python create_train_data.py --equation {equation} --size {size}'\
                        f' --N {shape - 1} --eps {epsilon} --kind {kind} --sd {sd} --forcing {forcing}', shell=True)
        data = LGDataset(equation=equation, pickle_file=file, shape=shape, kind=kind, sd=sd, forcing=forcing, transform_f=transform_f)
    return data


class LGDataset():
    """Legendre-Galerkin Dataset."""
    def __init__(self, equation, pickle_file, shape=64, transform_f=None, transform_a=None, kind='train', sd=1, forcing='uniform'):
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        if forcing == 'uniform':
            pickle_file += 'uniform'
        elif forcing == 'normal':
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
            ff = f.view(1, 1, self.shape)
            ff = self.transform_f(ff).view(1, self.shape)
            sample = {'u': u, 'f': f, 'a': a, 'p': p, 'fn': ff}
        else:
            sample = {'u': u, 'f': f, 'a': a, 'p': p}
        return sample


def normalize(gparams, loader):
    from torchvision import transforms
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for _, data in enumerate(loader):
        f = data['f']
        channels_sum += torch.mean(f, dim=[0, 2])
        channels_squares_sum += torch.mean(f**2, dim=[0,2])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches - mean**2)**0.5
    gparams['mean'] = float(mean[0].item())
    gparams['std'] = float(std[0].item())
    return gparams, transforms.Normalize(mean, std)

