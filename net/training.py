#training.py
import random
import torch
import network
from data_loader import LGDataset, show_solution
import numpy as np
import matplotlib.pyplot as plt


# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Load the dataset
lg_dataset = LGDataset(pickle_file='1000')

# N is batch size; D_in is input dimension; D_out is output dimension.
N, D_in, Filters, D_out = 128, 1, 1, 64

# Create random Tensors to hold inputs and outputs.
x = torch.Tensor([lg_dataset[:N]['u']]).reshape(N, 1, D_out)
y = torch.Tensor([lg_dataset[:N]['f']]).reshape(N, 1, D_out)
x,y = x.to(device), y.to(device)

# Construct our model by instantiating the class
model = network.Net(1, Filters, D_out)
model.to(device)

# Construct our loss function and an Optimizer.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
for t in range(100):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)
  # Compute and print loss
  loss = criterion(y_pred, y)
  print(f"Epoch: {t}, Loss: {loss.item()}")

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


yhat = y_pred[0,0,:].to('cpu').detach().numpy()
yhat = np.array(yhat)
target = np.array(lg_dataset[0]['f'])
sol = np.array(lg_dataset[0]['u'])
plt.figure(1, figsize=(10,6))
plt.grid(alpha=0.618)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Example')
plt.plot(yhat, label='Pred')
plt.plot(target, label='Target')
plt.plot(sol, label='Solution')
plt.legend(shadow=True)
plt.show()