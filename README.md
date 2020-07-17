# Legendre-Galerkin Deep Neural Network

This repository solves various differential equations using the Spectral Element Method, creates datasets, trains networks to find solutions and then validates the model.

### Training.py
This script trains a network and will create the dataset if needed, as well as log validation metrics.

Parameters:
`equation: Standard, Burgers, Helmholtz`
`model: NetA, ResNet`

Use
```
python training.py --equation Burgers --model NetA --loss MSE --blocks 3 --file 10000N63 --batch 25000 --epochs 2000 --ks 5 --filters 32 --nbfuncs 1
```

