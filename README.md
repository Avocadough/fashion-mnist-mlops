<div align="center">
  <h1>Fashion-MNIST MLOps</h1>
  <p>A coursework pipeline for Fashion-MNIST validation, model selection, MLflow tracking and FastAPI inference.</p>
  <p>
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square" alt="scikit-learn">
    <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square" alt="MLflow tracking">
    <img src="https://img.shields.io/badge/FastAPI-Inference-009688?style=flat-square" alt="FastAPI inference">
  </p>
  <p><a href="#workflow">Workflow</a> · <a href="#project-map">Project map</a> · <a href="#run-locally">Run locally</a> · <a href="#evaluation-limits">Evaluation limits</a></p>
</div>

## Workflow

1. Load Fashion-MNIST through OpenML and log validation and anomaly checks.
2. Scale pixels and create stratified training, validation and test splits.
3. Train PCA + logistic regression candidates, logging parameters and validation metrics to MLflow.
4. Select by validation accuracy, evaluate on the test split and export a model for the API.

The repository also includes a synthetic `--tiny` validation mode and GitHub Actions configuration. Tiny data are randomly generated for checks, not Fashion-MNIST or evidence of model accuracy.

## Project map

- [`scripts/`](scripts/) — validation, preprocessing, training and evaluation/export.
- [`app/main.py`](app/main.py) — FastAPI application with `/health`, `/predict` and `/predict_batch`.
- [`tests/`](tests/) — preprocessing, validation and API checks; some API tests use a dummy model.
- [`environment.yml`](environment.yml), [`requirements.txt`](requirements.txt) and [`Dockerfile`](Dockerfile) — environment and container definitions.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — configured CI using tiny synthetic data.

## Run locally

The supplied Conda environment specifies Python 3.11. From the repository root:

```bash
conda env create -f environment.yml
conda activate mlflow-fashion
python scripts/1_data_validation.py
python scripts/2_data_preprocessing.py
python scripts/3_model_training.py
python scripts/4_model_evaluation.py
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

The first step fetches data over the network; model training can take considerable resources. These commands describe the source and have not been rerun for this README. The evaluation/export script **replaces** `artifacts/deploy/fashion_mnist_clf` if it already exists; preserve anything needed there first. The application needs that exported model folder at startup. Open `/docs` for request schemas; `/predict` accepts 784 pixel values under `pixels`, and `/predict_batch` accepts a `batch` of such lists. Values above 1 trigger scaling by 255 for the request.

For validation without the download:

```bash
python scripts/1_data_validation.py --tiny
```

`--tiny` creates a validation report but does not write `artifacts/fashion_mnist.csv`; preprocessing therefore needs an existing CSV. The CI file contains a separate synthetic CSV creation step. Tests that inspect splits or reports require those artifacts, while API tests have their own model and dependency requirements. The configured CI is not a guarantee of a passing run.

## Evaluation limits

The test split measures the selected candidate on this dataset, not deployment performance or performance on photographs from a different distribution. The export and serving path should be verified in the intended environment. The threshold in the evaluation script is a reporting flag, not proof that a model is production-ready. The dataset is referenced as Fashion-MNIST via OpenML; check its terms separately before reuse.
