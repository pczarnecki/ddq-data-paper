# Data-Driven Quadrature for Longwave and Shortwave Absorption by Major Greenhouse Gases

This repository contains the information necessary to produce the quadrature scheme configurations and analysis in *Czarnecki and Brath, 2025* as follows:

- `DDQ configurations` stores the wavenumbers (cm$^{-1}$) and accompanying quadrature weights of the quadrature scheme for present-day longwave (`DDQ_LW_present`) and shortwave (`DDQ_SW_present`) calculations, as well as sets that are trained on variability in the concentrations of major greenhouse gases (`DDQ_LW` and `DDQ_SW`). Additional optimizations are available in the subfolder.
- `figures` contains the figures in *Czarnecki and Brath.* 
	- `Generate Figures.ipynb` creates these plots
- `Tutorial.ipynb` describes the full process of optimizing a quadrature scheme, including creating a training dataset of line-by-line fluxes with ARTS, writing the cost functions, setting up the training data and running the optimization procedure via Python package `datadrivenquadrature`.