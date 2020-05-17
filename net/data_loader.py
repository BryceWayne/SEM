#data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

class LGDataset():
    """Legendre-Galerkin Dataset."""

    def __init__(self, pickle_file):
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        with open('../data/' + pickle_file + '.pkl', 'rb') as f:
        	self.data = pickle.load(f)
        	self.data = self.data[:,:,0,:]
        	print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data[:,0,:][idx]
        u = self.data[:,1,:][idx]
        f = self.data[:,2,:][idx]
        sample = {'x': x, 'u': u, 'f': f}
        return sample


def show_solution(solution):
	plt.figure(1, figsize=(10,6))
	plt.title('Exact Solution')
	plt.plot(solution[0], solution[1], label='Exact')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.grid(alpha=0.618)
	plt.show()


face_dataset = LGDataset(pickle_file='1000')

for i in range(len(face_dataset)):
    sample = face_dataset[i]    
    show_solution([sample['x'], sample['u']])
    if i == 3:
        plt.show()
        break