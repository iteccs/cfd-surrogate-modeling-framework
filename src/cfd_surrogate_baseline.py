
"""
CFD Surrogate Modeling Framework

Author and Owner:
Dr. Ahmed Kaffel, PhD
ITECCS – Institute of Technology, Engineering, Computing, and Computational Sciences

If this software or an adapted version is used in research, reports,
presentations, theses, or publications, please acknowledge:

Kaffel, A. (2026). CFD Surrogate Modeling Framework. ITECCS.

License: MIT
"""

# cfd_surrogate_baseline.py
# Baseline CFD surrogate model:
# CFD inputs -> PCA-reduced CFD outputs -> reconstructed predictions
#
# Author: Ahmed Kaffel / ITECCS
#
# Purpose:
# This script provides a first data-driven baseline for CFD surrogate modeling.
# It uses PCA to reduce the output space and Ridge regression to learn the
# mapping from geometry/flow parameters to CFD output quantities.
#
# Expected CSV structure:
# One row = one simulation case
#
# Example input columns:
#   D, Lx, Ly, velocity_inlet, discharge, Reynolds, Froude
#
# Example output columns:
#   mean_velocity, max_velocity, mean_pressure, max_shear, free_surface_mean
#
# The user should adapt input_cols and output_cols to match the actual dataset.



import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# --------------------------------------------------
# 1. User settings
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Baseline CFD surrogate model using PCA and Ridge regression."
)

parser.add_argument(
    "--data",
    type=str,
    default="data/sample_cfd_dataset.csv",
    help="Path to the CFD dataset CSV file."
)

parser.add_argument(
    "--output",
    type=str,
    default="results/baseline_results",
    help="Directory where results will be saved."
)

args = parser.parse_args()

CSV_FILE = args.data
OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Input parameters: geometry + hydraulic/flow conditions
input_cols = [
    "D",
    "Lx",
    "Ly",
    "velocity_inlet",
    "discharge",
    "Reynolds",
    "Froude"
]

# Output quantities: processed CFD results
output_cols = [
    "mean_velocity",
    "max_velocity",
    "mean_pressure",
    "max_shear",
    "free_surface_mean"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# PCA keeps enough components to explain this percentage of variance
VARIANCE_TO_KEEP = 0.99


# --------------------------------------------------
# 2. Load and check data
# --------------------------------------------------

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find {CSV_FILE}. "
        "Please place the CFD dataset CSV file in the same folder as this script."
    )

df = pd.read_csv(CSV_FILE)

required_cols = input_cols + output_cols
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"The following required columns are missing from the dataset: {missing_cols}\n"
        "Please update input_cols and output_cols to match your CSV file."
    )

df = df[required_cols].dropna()

if len(df) < 5:
    raise ValueError(
        "The dataset contains too few complete cases after removing missing values. "
        "Please provide more simulation cases."
    )

print(f"Dataset loaded successfully: {df.shape[0]} complete simulation cases")
print(f"Number of input variables: {len(input_cols)}")
print(f"Number of output variables: {len(output_cols)}")


# --------------------------------------------------
# 3. Split inputs and outputs
# --------------------------------------------------

X = df[input_cols].values
Y = df[output_cols].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# 4. Scale outputs and apply PCA
# --------------------------------------------------

Y_scaler = StandardScaler()
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

pca_full = PCA()
pca_full.fit(Y_train_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.searchsorted(cumulative_variance, VARIANCE_TO_KEEP) + 1

print("\nPCA information")
print("----------------")
print(f"Number of PCA components kept: {n_components}")
print(f"Explained variance kept: {cumulative_variance[n_components - 1]:.4f}")

pca = PCA(n_components=n_components)
Y_train_pca = pca.fit_transform(Y_train_scaled)


# --------------------------------------------------
# 5. Train regression model
# --------------------------------------------------

model = Pipeline([
    ("X_scaler", StandardScaler()),
    ("regressor", RidgeCV(alphas=np.logspace(-6, 6, 50)))
])

model.fit(X_train, Y_train_pca)


# --------------------------------------------------
# 6. Predict and reconstruct outputs
# --------------------------------------------------

Y_pred_pca = model.predict(X_test)
Y_pred_scaled = pca.inverse_transform(Y_pred_pca)
Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

metrics = []

for i, col in enumerate(output_cols):
    rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
    mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
    r2 = r2_score(Y_test[:, i], Y_pred[:, i])

    metrics.append({
        "output": col,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

metrics_df = pd.DataFrame(metrics)
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nModel metrics")
print("-------------")
print(metrics_df)


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

results = pd.DataFrame()

for i, col in enumerate(input_cols):
    results[col] = X_test[:, i]

for i, col in enumerate(output_cols):
    results[col + "_true"] = Y_test[:, i]
    results[col + "_pred"] = Y_pred[:, i]

predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
results.to_csv(predictions_path, index=False)


# --------------------------------------------------
# 9. Save PCA information
# --------------------------------------------------

pca_info = pd.DataFrame({
    "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
    "explained_variance_ratio": pca_full.explained_variance_ratio_,
    "cumulative_variance": cumulative_variance
})

pca_path = os.path.join(OUTPUT_DIR, "pca_explained_variance.csv")
pca_info.to_csv(pca_path, index=False)


# --------------------------------------------------
# 10. Generate plots
# --------------------------------------------------

for i, col in enumerate(output_cols):
    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test[:, i], Y_pred[:, i])
    plt.xlabel(f"True {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"True vs Predicted: {col}")

    min_val = min(Y_test[:, i].min(), Y_pred[:, i].min())
    max_val = max(Y_test[:, i].max(), Y_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], "--")

    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"true_vs_pred_{col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


# --------------------------------------------------
# 11. Final message
# --------------------------------------------------

print("\nDone.")
print(f"All results were saved in: {OUTPUT_DIR}")
print(f"- Metrics: {metrics_path}")
print(f"- Predictions: {predictions_path}")
print(f"- PCA information: {pca_path}")
print("\nNext step:")
print("If the baseline results are reasonable, extend the model to a nonlinear neural network or autoencoder.")
