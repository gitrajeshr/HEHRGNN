# HEHRGNN

## Required Packages
The major libraries/packages required by HEHRGNN are listed below:

- torch
- torch-geometric
- torch_scatter

Other dependencies are assumed to be resolved automatically by pip. 

## Training and evaluation

The HEHRGNN model can be invoked for training and/or evaluation by running main.py. 

The example below illustrates the command line options  for invoking HEHRGNN in training and evaluation mode:

python main.py -dataset="wd50k_unified_format" -emb_dim=128 -num_gnn_layers=2 -epochs=50

The example below illustrates the command line options  for invoking HEHRGNN in only evaluation mode:

python main.py -dataset="wd50k_unified_format" -run_mode="eval" 