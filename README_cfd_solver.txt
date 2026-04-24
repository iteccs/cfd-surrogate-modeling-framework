CFD Surrogate Baseline Solver
=================================

Owner: Dr. Ahmed Kaffel

Description:
------------
This solver provides a baseline data-driven model for CFD simulations.
It uses:
- Principal Component Analysis (PCA) for output dimensionality reduction
- Ridge Regression for learning the mapping between CFD inputs and outputs

Purpose:
--------
The goal of this solver is to:
- Build a first surrogate model of CFD simulations
- Verify dataset consistency and predictive capability
- Provide a baseline for advanced modeling (e.g., neural networks)

Required Input:
---------------
A CSV file named:
    cfd_dataset.csv

Format:
- One row per simulation case
- Must include:
    Inputs:
        D, Lx, Ly, velocity_inlet, discharge, Reynolds, Froude
    Outputs:
        mean_velocity, max_velocity, mean_pressure, max_shear, free_surface_mean

You may modify column names in the script if needed.

How to Run:
-----------
1. Place the dataset (cfd_dataset.csv) in the same folder
2. Run:
       python cfd_surrogate_baseline.py

Outputs:
--------
A folder named:
    baseline_results/

Containing:
- metrics.csv (model performance)
- predictions.csv (true vs predicted values)
- pca_explained_variance.csv
- plots for each output variable

Notes:
------
- This is a baseline model, not the final model
- Performance depends primarily on dataset quality
- If results are poor, improve dataset consistency and coverage

Next Steps:
-----------
- Extend to nonlinear models (PyTorch)
- Use autoencoders instead of PCA
- Incorporate physics-based constraints

Contact:
--------
For questions or collaboration:
Dr. Ahmed Kaffel
ITECCS | University of Wisconsin–Milwaukee
