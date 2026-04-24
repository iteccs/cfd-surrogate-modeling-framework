**Dr. Ahmed Kaffel**  
ITECCS | University of Wisconsin–Milwaukee  

A research-oriented framework for data-driven surrogate modeling of CFD simulations using reduced-order modeling and machine learning.


## Overview
This framework provides a structured and research-oriented approach to building surrogate models for CFD simulations. It enables the construction of reduced-order representations of complex flow systems, allowing efficient prediction of key physical quantities from geometric and flow parameters.

The goal is to bridge high-fidelity CFD simulations with data-driven modeling, forming a foundation for scalable, generalizable, and publication-quality computational frameworks.

## Methodology
- PCA for dimensionality reduction  
- Ridge regression for mapping inputs to outputs  

## Usage
```bash
python cfd_surrogate_baseline.py


## Input Data
- CSV file with one row per simulation case
- Must include geometry, flow parameters, and CFD outputs

## Outputs
- metrics.csv (performance)
- predictions.csv (true vs predicted)
- plots (visual comparisons)
