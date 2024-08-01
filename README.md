# Subgradient Langevin Methods for Sampling from Non-smooth Potentials
Source code to reproduce the results of the paper "Subgradient Langevin Sampling for Non-smooth Potentials", by A. Habring, M. Holler, T. Pock


## Dependencies
We recommend to use conda to create an environment satisfying the necessary dependencies. Navigating to the current directory in the terminal simply run:
```
conda env create -f environment.yml
```
Part of the repository includes Jupyter Notebooks. To use the environment also in your Jupyter notebook run
```
python -m ipykernel install --user --name=subgradient_langevin
```
YOu should then be able to choose the environment by selecting the respective kernel in the Jupyter notebook.

## Reproducing the results
### Sampling
#### 2d examples
Sampling according to the 2D examples from section 6.1 in the paper is done by running the files 2d_example.py and 2d_example_l1.py. At the top of the two scripts you find several parameters which are currently set to the values used in the paper but can be adapted.

#### Imaging examples
The imaging examples are reproduced by running the files image_***.py where *** denotes the respective experiment to reproduce.

The Belief propagation code is contained in belief_propagation.ipynb.

### Visualization
The visualization/plots are computed in the Jupyter notebooks evaluate_imaging.ipynb and evaluate_2d.ipynb.