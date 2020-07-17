# Legendre-Galerkin Deep Neural Network

This repository solves various differential equations using the Spectral Element Method, creates datasets, trains networks to find solutions and then validates the model.

### Training.py
This script trains a network and will create the dataset if needed, as well as log validation metrics.

Parameters:

Differential Equation: `equation`**:**`Standard, Burgers, Helmholtz`

NN Architecture: `model`**:**`ResNet, NetA`

Num. of Blocks: `blocks`**:**`0,1,2,3,...`

Loss Func,: `loss`**:**`MAE, MSE`

Dataset: `file`**:**`1000N127` (Example)

Batch Size: `batch`**:**`1000` (Example)

Num. of Epochs: `epochs`**:**`50000` (Example)

Kernel Size: `ks`**:**`5` (Example) *Should be odd

Num. of Filters: `filters`**:**`32` (Example)

Num. of Basis Functions: `nbfuncs`**:**`0,1,2,3,...` (Example)

Use
```
python training.py --equation Burgers --model NetA --loss MSE --blocks 3 --file 10000N63 --batch 25000 --epochs 2000 --ks 5 --filters 32 --nbfuncs 1
```

