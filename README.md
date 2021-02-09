# Bidirectional Graph Convolutional Networks

This is the code repository of *Make GCNs Work without Features: Joint Learning of Features and Labels for Partially
Observed Graphs*, submitted to KDD 2021.

## Requirements

All codes are implemented by Python 3.7 with the requirements listed in `requirements.txt`.
```
pip install -r requirements.txt
``` 

You should install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), which is a package that our
GCN model is implemented on. The GitHub site of package describes well how to install the package in various settings,
and so we recommend following the instructions. 

## Datasets

The datasets are automatically downloaded based on the torch-geometric package.

## Usage

Simply running the following command will trains our BiGCN model for the datasets. You can change the hyperparameters or
other environments of experiments by modifying `main.py`. 
```
bash main.sh
```
