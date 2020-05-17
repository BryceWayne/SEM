#data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt


class LGDataset(Dataset):
    """Legendre-Galerkin Dataset."""

    def __init__(self, pickle_file):
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.data = load_obj(pickle_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data
        u = self.data
        sample = {'x': x, 'solution': u}
        return sample

    def load_obj(name):
	    with open(name + '.pkl', 'rb') as f:
	        return pickle.load(f)


def show_solution(solution):
	plt.figure(1, figsize=(10,6))
	plt.title('Exact Solution')
	plt.plot(solution, label='Exact')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend(shadow=True)
	plt.grid(alpha=0.618)
	plt.show()


face_dataset = FaceLandmarksDataset(pickle_file='data/test.pkl')

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break