# fraud-detection-ml-system

An example end-to-end fraud detection machine learning project (data, notebooks, models, and a lightweight app). Designed to be demo-ready for GitHub and Docker-based showcase.

## Highlights

- **Notebooks**: multiple interactive notebooks for exploration and model experiments (`notebook/`).
- **Data**: example dataset at `data/input/creditcard.csv` for reproducing experiments.
- **Code**: lightweight app and modular code under `src/` for inference and utilities.
- **Docker-ready**: `Dockerfile` and `compose.yaml` included for quick demos.

## Quickstart (local)

1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the app (example):

```powershell
python src\app.py
```

3. Open notebooks:

```powershell
jupyter lab
```

Then open one of the notebooks in `notebook/` (for example `fraud-detection.ipynb`) to explore data and training steps.

## Docker (demo)

Build the image and run the container:

```powershell
docker build -t fraud-detection .
docker run --rm -p 8000:8000 fraud-detection
```

Adjust the port mapping to your app's configuration if necessary.

## Project Structure

- `src/` — application entrypoint and package modules (`fraud_detection/`).
- `notebook/` — exploratory analysis and model training notebooks.
- `data/` — example datasets (`data/input/creditcard.csv`).
- `models/` — trained model artifacts (if present) and model outputs.
- `Dockerfile`, `compose.yaml` — container and compose definitions for demos.

## Tech & Tools (used in notebooks)

Below are the main libraries, frameworks, and techniques used across the notebooks in `notebook/` — include these on your GitHub README to highlight your data-science stack and skills.

- **Data processing:** `pandas`, `polars`, `numpy`
- **Modeling / ML frameworks:** `scikit-learn` (models, metrics, model selection), `jax` + `jax.numpy` for custom neural nets
- **Optimization & training:** `optuna` (hyperparameter search, importance), `optax` (optimizers for JAX models)
- **Algorithms showcased:** Decision Trees, Random Forests, neural networks (ANNs implemented with JAX)
- **Evaluation & calibration:** `sklearn.metrics` (precision, recall, average-precision/AP-score, ROC/AUC), `sklearn.calibration` (calibration curves)
- **Visualization:** `matplotlib`, `plotly` (interactive figures)
- **Utilities:** `tqdm` (progress bars), `pathlib`, Python `typing` utilities
- **Experiment & model analysis techniques:** EDA, feature importance analysis, hyperparameter tuning (including multi-objective tuning), calibration analysis, and train/test splitting and cross-validation

Add these keywords to your README or repository description to clearly show the practical tools and techniques you used in the project.

## Data Science Skills

This project demonstrates the following practical data-science skills — paste these as a focused skills section on your GitHub profile or README to highlight your expertise:

- **Modeling & ML engineering:** Implemented custom neural networks using `jax`/`jax.numpy` with JIT-compiled model functions and manual parameter initialization.
- **Neural network training:** Built training loops with `optax` (Adam, learning-rate schedules), gradient calculation via `jax.value_and_grad`, and recording training history.
- **Imbalanced data handling:** Applied class-weighted binary cross-entropy, stratified `train_test_split`, and threshold tuning to handle extreme class imbalance common in fraud detection.
- **Calibration & uncertainty:** Implemented temperature-scaling for probability calibration, computed NLL from logits, and produced calibration curves to assess probabilistic accuracy.
- **Evaluation & metrics:** Extensive use of precision–recall curves, ROC/AUC, average-precision (AP), confusion matrices, and F1-based threshold selection for business-focused evaluation.
- **Feature engineering & data pipelines:** Fast ingestion with `polars` and `pandas`, feature scaling with a JAX-compatible scaler, and reproducible train/validation/test splits.
- **Hyperparameter tuning & analysis:** Experimentation and hyperparameter search using `optuna`, plus parameter importance analysis and multi-objective tuning patterns.
- **Visualization & storytelling:** Interactive and publication-ready plots with `plotly` (loss curves, PR/ROC, calibration) and supporting Matplotlib figures for static reports.
- **Performance & reproducibility:** Use of `@jax.jit` for speed, typed code (`TypedDict`/`typing`), progress indicators (`tqdm.notebook`), and instructions for Docker-based reproducible demos.

Use this concise skill list in your README or GitHub profile to clearly advertise the practical, production-minded data-science competencies demonstrated by the notebooks.

## Fraud Detection Approaches

The `notebook/fraud-detection-AE-ANN.ipynb` notebook showcases advanced anomaly detection techniques to identify fraudulent transactions:

### Approach A - Isolation Forest
A baseline unsupervised approach using `scikit-learn`'s **Isolation Forest**. This tree-based ensemble method explicitly isolates anomalies by randomly selecting a feature and a split value. It is efficient for high-dimensional datasets and serves as a strong benchmark for fraud detection.

### Approach B - Deep Autoencoder (JAX/Optax)
A sophisticated unsupervised Neural Network implementation using **JAX** and **Optax**.

*   **Architecture**: Symmetric Autoencoder designed to learn the compressed representation of normal transactions.
    *   Structure: `Input (29) -> Hidden (16) -> Bottleneck (8) -> Hidden (16) -> Output (29)`
*   **Technology**:
    *   **JAX**: Leveraged for high-performance numerical computing and Just-In-Time (JIT) compilation (`@jax.jit`) to speed up model training and inference.
    *   **Optax**: Used for the optimization loop, employing the **AdamW** optimizer with a Warmup Cosine Decay learning rate schedule for stable convergence.
*   **Methodology**:
    *   **Training**: The model is trained exclusively on **non-fraudulent (normal)** transactions to learn the manifold of "legitimate" behavior.
    *   **Detection**: Anomalies (fraud) are detected by calculating the **Reconstruction Error** (Mean Absolute Error) between the input and the reconstructed output. High reconstruction errors indicate outliers that deviate from the learned normal patterns.

## Reproduce Experiments

- Install dependencies from `requirements.txt`.
- Open and run the notebooks in `notebook/` to follow the EDA, preprocessing, training, and evaluation steps.
- The notebooks show end-to-end examples and are the primary entrypoint for reproducing results.

## Show it off on GitHub

- Add screenshots or an animated GIF of the notebook output or the app in action (place under `docs/` or `assets/`).
- Add a short demo video or link in the README to highlight model performance and UX.
- Add CI badges (GitHub Actions) and a license badge for a professional touch.

Example commands to create a new GitHub repo and push:

```powershell
git init; git add .; git commit -m "Initial commit"; gh repo create <your-username>/fraud-detection-ml-system --public --source=. --remote=origin; git push -u origin main
```

Replace `<your-username>` with your GitHub handle. You can also add a `README` screenshot and GitHub Pages or a repo description to increase discoverability.

## Development & Contributing

- Use the notebooks for rapid prototyping. When an experiment becomes stable, extract training code into `src/fraud_detection/models/` for reuse.
- Open issues and PRs for bugfixes, feature requests, and documentation improvements.

## Data & License

- Example dataset included for demo/educational purposes: `data/input/creditcard.csv`.
- Add or update a `LICENSE` file to reflect how you want to share the project.

## Next steps / Suggestions

- Add a short `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for community contributions.
- Add GitHub Actions workflow for linting and notebook testing to show CI status.
- Include a few annotated screenshots and a short demo GIF in `docs/` to make the README visually appealing on GitHub.

---

If you'd like, I can also:

- add badges (build, license),
- create a `docs/` folder and add a demo GIF, or
- wire up a basic GitHub Actions workflow for CI. Which would you prefer next?
