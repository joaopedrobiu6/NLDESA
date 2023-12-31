# NLDESA - Nonlinear Differential Equation Stability Analysis

This is a Python package for the stability analysis of nonlinear differential equations using JAX [1] and pyDMD [2] [3].
The package is still in development and is not yet ready for use.

Authors: João Biu (joaopedrofbiu@tecnico.ulisboa.pt), Miguel Lameiras (miguel.lameiras@tecnico.ulisboa.pt)

[1] https://github.com/google/jax

[2] Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530, https://doi.org/10.21105/joss.00530 (https://github.com/PyDMD/PyDMD.git)

## Installation
Using pip:
`pip install nldesa`

For developers
- Clone the repository
`cd NLDESA && pip install -e .`

## Stability Map Zoom

Each yellow pixel in the plot corresponds to a "stable" solution of the non linear Mathieu Equation. Each frame has 350*350 solutions computed in around 190 seconds.

![Stability Map Zoom](docs/FirstEverZoom.gif)

Video with full resolution availabe in the [docs](https://github.com/joaopedrobiu6/NLDESA/docs/FirstEverZoom.gif)!
