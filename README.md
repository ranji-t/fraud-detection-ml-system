# Fraud Detection ML System

> An end-to-end machine learning system for credit card fraud detection — from raw data and exploratory analysis through production-grade, containerised inference. Built to demonstrate the full data science lifecycle: rigorous EDA, classical and deep-learning models, custom metrics, hyperparameter optimisation, model explainability, and CI/CD-enforced code quality.

[![CI](https://github.com/ranjeethrp15/fraud-detection-ml-system/actions/workflows/quality_check_flow.yml/badge.svg)](https://github.com/ranjeethrp15/fraud-detection-ml-system/actions/workflows/quality_check_flow.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Ruff](https://img.shields.io/badge/linter-ruff-orange)

---

## What's Inside

Five notebooks and a modular Python package that walk through every stage of a real fraud detection problem — a heavily imbalanced, high-stakes binary classification task on the canonical [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

| Notebook | Focus |
|---|---|
| `fraud-detection.ipynb` | EDA, feature engineering, classical baselines (Decision Tree, Random Forest, Logistic Regression) |
| `fd-lr-jax.ipynb` | Logistic Regression implemented **from scratch** in JAX with JIT-compiled training |
| `fd-ann-jax.ipynb` | Artificial Neural Network built **from scratch** in JAX with manual backpropagation via `jax.value_and_grad` |
| `fraud-detection-AE-ANN.ipynb` | Unsupervised anomaly detection: Isolation Forest baseline vs. a **Deep Autoencoder** (JAX + Optax) |
| `fraud-detection-MOO.ipynb` | **Multi-Objective Optimisation** with Optuna — simultaneously optimising recall and precision |

---

## Data Science Skills Demonstrated

### Modelling & ML Engineering
- Implemented Logistic Regression and multi-layer ANNs **from scratch** in JAX — no Keras, no PyTorch; raw matrix ops, manual weight initialisation, and JIT-compiled forward passes
- Built a symmetric Deep Autoencoder (`Input(29) → 16 → Bottleneck(8) → 16 → Output(29)`) for unsupervised fraud detection via reconstruction error
- Trained XGBoost gradient-boosted trees alongside JAX neural nets for direct comparison
- Used **Flax** for structured, higher-level neural network modules on top of JAX

### Handling Class Imbalance
- Applied class-weighted binary cross-entropy loss to handle extreme imbalance (~0.17% fraud rate)
- Implemented stratified train/validation/test splits to preserve class ratios across all partitions
- Performed threshold tuning using F1-optimal cutoff selection from precision–recall curves

### Hyperparameter Optimisation
- Used **Optuna** for automated hyperparameter search with trial pruning
- Ran **Multi-Objective Optimisation (MOO)** — Pareto-front search simultaneously trading off precision vs. recall, reflecting real-world business constraints (cost of false positives vs. false negatives)
- Analysed parameter importance to understand which hyperparameters drive model performance

### Custom Evaluation Metrics
Wrote production-style, fully validated custom metrics in `src/fraud_detection/clf/metrics/`:
- **Precision@K** — fraction of true frauds in the top-K highest-risk predictions
- **Recall@K** — fraction of all frauds captured in the top-K predictions
- **Lift@K** — how much better the model performs than random selection at rank K

These metrics reflect how fraud models are *actually evaluated* in operations — not just AUC.

### Model Explainability
- Used **SHAP** (SHapley Additive exPlanations) for model-agnostic feature importance
- Produced summary plots and individual prediction explanations to make black-box models interpretable

### Probability Calibration
- Applied temperature scaling for post-hoc probability calibration
- Computed Negative Log-Likelihood (NLL) from logits and generated calibration curves to verify probabilistic accuracy

### Evaluation & Visualisation
- Precision–Recall curves, ROC/AUC, Average Precision (AP), confusion matrices
- Interactive Plotly figures (loss curves, PR/ROC curves, calibration plots)
- Publication-quality Matplotlib static figures

### Data Engineering
- Fast data ingestion with **Polars** (lazy evaluation, columnar) and **Pandas** for compatibility
- **PyArrow** for efficient columnar storage and interchange
- **FastExcel** for high-performance Excel I/O
- JAX-compatible feature scaling pipeline

### Software Engineering & MLOps
- **Type-safe configuration** — all model and training parameters defined as `Pydantic` `BaseModel` subclasses with field validators; YAML configs loaded via **OmegaConf**
- **CI/CD pipeline** via GitHub Actions: runs `ruff check`, `ruff format`, and `pytest` on every push and PR to `main`/`dev`
- **Locked, reproducible dependencies** via `uv` and `uv.lock`
- **Docker** containerisation (`Dockerfile` + `compose.yaml`) for fully reproducible demos
- **nbstripout** to keep notebook output out of version control — clean diffs
- Python **3.13**, managed with `uv`

---

## Tech Stack

| Category | Tools |
|---|---|
| Neural networks | JAX, Flax, Optax |
| Classical ML | scikit-learn, XGBoost |
| Hyperparameter search | Optuna (single- and multi-objective) |
| Explainability | SHAP |
| Data processing | Polars, Pandas, PyArrow, FastExcel |
| Visualisation | Plotly, Matplotlib |
| Configuration | Pydantic, OmegaConf |
| Package management | uv |
| Code quality | Ruff (lint + format), pytest |
| CI/CD | GitHub Actions |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
fraud-detection/
├── notebook/                      # Five end-to-end experiment notebooks
│   ├── fraud-detection.ipynb      # EDA + classical baselines
│   ├── fd-lr-jax.ipynb            # Logistic Regression from scratch (JAX)
│   ├── fd-ann-jax.ipynb           # ANN from scratch (JAX)
│   ├── fraud-detection-AE-ANN.ipynb  # Unsupervised: Isolation Forest + Autoencoder
│   └── fraud-detection-MOO.ipynb  # Multi-objective hyperparameter optimisation
├── src/
│   └── fraud_detection/
│       ├── clf/
│       │   ├── metrics/           # Custom metrics: Precision@K, Recall@K, Lift@K
│       │   └── utils/plotting/    # Reusable plotting utilities
│       └── config/
│           ├── config.py          # Pydantic-validated training config models
│           ├── load_config.py     # OmegaConf YAML loader
│           └── unsupervised/      # Autoencoder-specific config schema
├── config/                        # YAML configuration files
│   ├── base.yaml
│   └── autoencoder/autoencoder.yaml
├── data/
│   ├── input/creditcard.csv       # Kaggle credit card fraud dataset
│   └── db/optuna-fraud-detection.db  # Optuna study database
├── test/                          # pytest test suite
├── .github/workflows/             # GitHub Actions CI pipeline
├── Dockerfile
├── compose.yaml
└── pyproject.toml                 # uv-managed dependencies
```

---

## Fraud Detection Approaches

### Supervised Learning (Notebooks 1–4)

**Classical baselines** — Decision Tree, Random Forest, Logistic Regression evaluated with PR/ROC curves and threshold-optimised F1.

**JAX from scratch** — Logistic Regression and ANN implemented with raw JAX ops. Training loops use `jax.value_and_grad` for gradient computation, `optax` for Adam/scheduled optimisers, and `@jax.jit` for compiled execution — no ML framework abstractions.

**XGBoost** — Gradient-boosted trees with SHAP explainability.

### Unsupervised Anomaly Detection (Notebook: `fraud-detection-AE-ANN.ipynb`)

**Isolation Forest** — scikit-learn tree-based ensemble that isolates anomalies via random feature/split selection. Fast and parameter-light; used as an unsupervised baseline.

**Deep Autoencoder (JAX + Optax)** — Trains exclusively on *legitimate* transactions to learn the manifold of normal behaviour. At inference, high reconstruction error (MAE between input and output) flags fraud. Key implementation details:
- Warmup Cosine Decay learning-rate schedule via `optax`
- AdamW optimiser with weight decay
- JIT-compiled training step with `@jax.jit`
- Mini-batch training with reproducible shuffling

### Multi-Objective Optimisation (Notebook: `fraud-detection-MOO.ipynb`)

Optuna Pareto-front search trades off **Recall** (catching frauds) vs **Precision** (minimising false alarms) — directly reflecting the business cost structure of fraud operations, where both missed fraud and customer friction from false positives have real monetary cost.

---

## Quickstart

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```powershell
# Install dependencies
uv sync --all-extras --dev

# Launch notebooks
uv run jupyter lab
```

Open any notebook in `notebook/` to reproduce experiments end-to-end.

**Run the app:**

```powershell
uv run python src\app.py
```

**Run tests and linting:**

```powershell
uv run pytest
uv run ruff check src/ notebook/ test/
uv run ruff format --check src/ notebook/ test/
```

---

## Docker

```powershell
docker build -t fraud-detection .
docker run --rm -p 8000:8000 fraud-detection
```

Or with Compose:

```powershell
docker compose up
```

---

## CI/CD

Every push and pull request to `main` or `dev` triggers the GitHub Actions pipeline:

1. **Ruff lint** — enforces code style and catches common errors
2. **Ruff format check** — consistent formatting across `src/`, `notebook/`, `test/`
3. **pytest** — runs the full test suite

Dependencies are installed from `uv.lock` for a fully reproducible CI environment.

---

## Data

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds (0.172%). PCA-transformed features V1–V28, plus `Time`, `Amount`, and `Class`.
