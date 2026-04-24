"""
CFD Surrogate Modeling Framework

Author and Owner:
Dr. Ahmed Kaffel, PhD
ITECCS – Institute of Technology, Engineering, Computing, and Computational Sciences

Description:
This software provides a baseline data-driven framework for CFD surrogate modeling.
It maps geometric and flow parameters to reduced-order CFD output representations
using Principal Component Analysis (PCA) and regularized regression.

Scientific Purpose:
This solver is intended as a first diagnostic and baseline model for evaluating
whether a CFD dataset is sufficiently structured, consistent, and predictive for
surrogate modeling. It is not intended to replace high-fidelity CFD directly, but
to establish a reproducible reduced-order modeling pipeline that can later be
extended to neural networks, autoencoders, or physics-informed models.

Expected Dataset:
A CSV file with one row per simulation case.

Example input columns:
    D, Lx, Ly, velocity_inlet, discharge, Reynolds, Froude

Example output columns:
    mean_velocity, max_velocity, mean_pressure, max_shear, free_surface_mean

Acknowledgment:
If this software or an adapted version is used in research, reports, presentations,
theses, or publications, please acknowledge:

Kaffel, A. (2026). CFD Surrogate Modeling Framework. ITECCS.

Suggested acknowledgment:
The authors acknowledge the use of the CFD Surrogate Modeling Framework developed
by Dr. Ahmed Kaffel, ITECCS, for baseline reduced-order and data-driven CFD
surrogate modeling.

License: MIT
"""

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
# 1. Command-line settings
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description=(
        "Baseline CFD surrogate model using PCA for reduced-order output "
        "representation and Ridge regression for prediction."
    )
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
    help="Directory where model outputs and plots will be saved."
)

parser.add_argument(
    "--variance",
    type=float,
    default=0.99,
    help="Cumulative PCA variance to retain. Default: 0.99."
)

parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="Fraction of data reserved for testing. Default: 0.2."
)

parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility. Default: 42."
)

args = parser.parse_args()

CSV_FILE = args.data
OUTPUT_DIR = args.output
VARIANCE_TO_KEEP = args.variance
TEST_SIZE = args.test_size
RANDOM_STATE = args.random_state

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Define input and output variables
# --------------------------------------------------

# Input parameters: geometry + hydraulic/flow conditions.
# Modify these names to match the CFD dataset.
input_cols = [
    "D",
    "Lx",
    "Ly",
    "velocity_inlet",
    "discharge",
    "Reynolds",
    "Froude"
]

# Output quantities: processed CFD results.
# Modify these names to match the CFD dataset.
output_cols = [
    "mean_velocity",
    "max_velocity",
    "mean_pressure",
    "max_shear",
    "free_surface_mean"
]


# --------------------------------------------------
# 3. Load and validate dataset
# --------------------------------------------------

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find the dataset file: {CSV_FILE}\n"
        "Please provide a valid CSV file using the --data option.\n"
        "Example: python src/cfd_surrogate_baseline.py --data data/cfd_dataset.csv"
    )

df = pd.read_csv(CSV_FILE)

required_cols = input_cols + output_cols
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"The following required columns are missing from the dataset: {missing_cols}\n"
        "Please update input_cols and output_cols in the script, or rename the CSV columns."
    )

df = df[required_cols].copy()
initial_rows = len(df)
df = df.dropna()
removed_rows = initial_rows - len(df)

if len(df) < 5:
    raise ValueError(
        "The dataset contains too few complete simulation cases after removing missing values.\n"
        "Please provide more complete CFD simulation cases."
    )

print("\nCFD Surrogate Modeling Framework")
print("--------------------------------")
print(f"Dataset file: {CSV_FILE}")
print(f"Complete simulation cases: {df.shape[0]}")
print(f"Rows removed due to missing values: {removed_rows}")
print(f"Number of input variables: {len(input_cols)}")
print(f"Number of output variables: {len(output_cols)}")


# --------------------------------------------------
# 4. Split inputs and outputs
# --------------------------------------------------

X = df[input_cols].values
Y = df[output_cols].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training cases: {X_train.shape[0]}")
print(f"Testing cases: {X_test.shape[0]}")


# --------------------------------------------------
# 5. Scale outputs and apply PCA
# --------------------------------------------------

Y_scaler = StandardScaler()
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

pca_full = PCA()
pca_full.fit(Y_train_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.searchsorted(cumulative_variance, VARIANCE_TO_KEEP) + 1
n_components = min(n_components, len(output_cols))

print("\nPCA information")
print("----------------")
print(f"Target cumulative variance: {VARIANCE_TO_KEEP:.4f}")
print(f"Number of PCA components kept: {n_components}")
print(f"Explained variance retained: {cumulative_variance[n_components - 1]:.4f}")

pca = PCA(n_components=n_components)
Y_train_pca = pca.fit_transform(Y_train_scaled)


# --------------------------------------------------
# 6. Train regression model
# --------------------------------------------------

model = Pipeline([
    ("X_scaler", StandardScaler()),
    ("regressor", RidgeCV(alphas=np.logspace(-6, 6, 50)))
])

model.fit(X_train, Y_train_pca)

selected_alpha = model.named_steps["regressor"].alpha_

print("\nRegression model")
print("----------------")
print("Model type: Ridge regression with cross-validated regularization")
print(f"Selected alpha: {selected_alpha}")


# --------------------------------------------------
# 7. Predict and reconstruct outputs
# --------------------------------------------------

Y_pred_pca = model.predict(X_test)
Y_pred_scaled = pca.inverse_transform(Y_pred_pca)
Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)


# --------------------------------------------------
# 8. Evaluate model
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
# 9. Save predictions
# --------------------------------------------------

results = pd.DataFrame()

for i, col in enumerate(input_cols):
    results[col] = X_test[:, i]

for i, col in enumerate(output_cols):
    results[col + "_true"] = Y_test[:, i]
    results[col + "_pred"] = Y_pred[:, i]
    results[col + "_error"] = Y_test[:, i] - Y_pred[:, i]

predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
results.to_csv(predictions_path, index=False)


# --------------------------------------------------
# 10. Save PCA information
# --------------------------------------------------

pca_info = pd.DataFrame({
    "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
    "explained_variance_ratio": pca_full.explained_variance_ratio_,
    "cumulative_variance": cumulative_variance
})

pca_path = os.path.join(OUTPUT_DIR, "pca_explained_variance.csv")
pca_info.to_csv(pca_path, index=False)


# --------------------------------------------------
# 11. Save run summary
# --------------------------------------------------

summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("CFD Surrogate Modeling Framework\n")
    f.write("================================\n\n")
    f.write("Author and Owner: Dr. Ahmed Kaffel, PhD\n")
    f.write("ITECCS – Institute of Technology, Engineering, Computing, and Computational Sciences\n\n")
    f.write(f"Dataset file: {CSV_FILE}\n")
    f.write(f"Complete simulation cases: {df.shape[0]}\n")
    f.write(f"Rows removed due to missing values: {removed_rows}\n")
    f.write(f"Training cases: {X_train.shape[0]}\n")
    f.write(f"Testing cases: {X_test.shape[0]}\n")
    f.write(f"Input variables: {input_cols}\n")
    f.write(f"Output variables: {output_cols}\n")
    f.write(f"PCA variance target: {VARIANCE_TO_KEEP}\n")
    f.write(f"PCA components retained: {n_components}\n")
    f.write(f"Explained variance retained: {cumulative_variance[n_components - 1]:.4f}\n")
    f.write(f"Selected Ridge alpha: {selected_alpha}\n\n")
    f.write("Acknowledgment:\n")
    f.write("If this software or an adapted version is used in research, reports,\n")
    f.write("presentations, theses, or publications, please acknowledge:\n\n")
    f.write("Kaffel, A. (2026). CFD Surrogate Modeling Framework. ITECCS.\n")


# --------------------------------------------------
# 12. Generate plots
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
# 13. Final message
# --------------------------------------------------

print("\nDone.")
print(f"All results were saved in: {OUTPUT_DIR}")
print(f"- Metrics: {metrics_path}")
print(f"- Predictions: {predictions_path}")
print(f"- PCA information: {pca_path}")
print(f"- Run summary: {summary_path}")
print("\nNext step:")
print("If the baseline results are reasonable, extend the model to a nonlinear neural network,")
print("autoencoder-based reduced-order model, or physics-informed surrogate model.")
