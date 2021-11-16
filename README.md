# Small, correlated changes in synaptic connectivity may facilitate rapid motor learning

Corresponding paper: [Feulner et al. 2021](https://www.biorxiv.org/content/10.1101/2021.10.01.462728v1)

## Requirements
Python 3 and PyTorch 1.8

## Usage
1) train a modular recurrent neural network to produce hand trajectories
	- run_initial.py
2) simulate visuomotor rotation experiment
	- run_perturbation.py
	- plasticity mode can be either upstream of motor cortices ("extrainput") or within motor cortices ("noupstream")

Model definition can be found in toolbox_pytorch.py

Scripts to produce illustrations of trained model similar to Figure 3-6 of the paper can be found as well.

## License
[MIT](https://choosealicense.com/licenses/mit/)
