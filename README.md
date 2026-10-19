# End-to-End MLOps: Heart Disease Classification

> Assignment starter implementation. Run the workflow yourself, inspect all metrics, and personalize the analysis and report before submission. This is an educational ML demo, not a medical diagnostic tool.

## What is included
- UCI Heart Disease dataset download and data preparation
- EDA plots and summary statistics
- Logistic Regression and Random Forest pipelines with stratified cross-validation
- MLflow experiment logging
- Reusable preprocessing + model pipeline saved with joblib
- FastAPI `/predict`, `/health`, and `/metrics` endpoints
- Pytest unit/API tests
- GitHub Actions CI workflow
- Dockerfile and Kubernetes deployment/service manifests
- Report template and screenshot checklist

## 1. Requirements
- Python 3.11
- Git
- Docker Desktop (for container steps)
- Optional: `kubectl` and Minikube or Docker Desktop Kubernetes

## 2. Local setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use `.\.venv\Scripts\activate.bat` from Command Prompt instead.

## 3. Download the dataset
```powershell
python -m src.data --download
```
This uses the `ucimlrepo` package to fetch UCI Heart Disease dataset ID 45 and writes a CSV to `data/heart.csv`. An internet connection is required for the first download. Review the dataset license/terms and cite the UCI repository in your report.

## 4. Run EDA and train
```powershell
python -m src.eda
python -m src.train
```
Outputs:
- `reports/figures/` — EDA and evaluation plots
- `reports/metrics.json` — held-out metrics for the selected model
- `reports/cv_results.json` — cross-validation summary
- `models/heart_disease_pipeline.joblib` — fitted preprocessing/model pipeline
- `models/model_metadata.json` — feature order and model details
- `mlruns/` — local MLflow tracking data

Open the MLflow UI in a second terminal:
```powershell
mlflow ui --backend-store-uri ./mlruns
```
Then visit `http://127.0.0.1:5000`.

Training uses a stratified train/test split and 5-fold stratified cross-validation. Model selection is based on mean CV ROC-AUC. Do not report scores until you have run the code; results can vary with dependency versions and dataset revisions.

## 5. Run tests and lint
```powershell
ruff check .
pytest -q
```

## 6. Start the API locally
Train the model first, then:
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs` for interactive API docs.
- `GET /health`
- `GET /metrics`
- `POST /predict`

Example request (feature names must match `models/model_metadata.json`):
```json
{
  "features": {
    "age": 57,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }
}
```
The endpoint returns a model class and class probabilities. The example values are illustrative only and are not medical advice. Use the exact feature list in the generated metadata if the upstream dataset has different column labels.

## 7. Docker build and run
```powershell
docker build -t heart-mlops-api:local .
docker run --rm -p 8000:8000 heart-mlops-api:local
```
Then test `/health`, `/docs`, and `/predict`. Model artifacts must exist before building the image; `models/` is included in the Docker build context.

## 8. Kubernetes (local Minikube example)
First build the image and make it visible to Minikube:
```powershell
minikube start
minikube image build -t heart-mlops-api:local .
kubectl apply -f deployment/k8s.yaml
kubectl rollout status deployment/heart-mlops-api
kubectl get pods,svc
kubectl port-forward service/heart-mlops-api 8000:80
```
Test `http://127.0.0.1:8000/health`. For a cloud cluster, push the image to a registry you control and change `image:` in `deployment/k8s.yaml`. Configure credentials/secrets securely; do not commit cloud credentials.

## 9. GitHub Actions
Create a GitHub repository, add these files, and push to the default branch. The workflow in `.github/workflows/ci.yml` installs dependencies, runs Ruff and pytest, downloads the dataset, trains the models, and uploads generated reports/model files as workflow artifacts. GitHub-hosted runners need network access to fetch the UCI dataset.

## 10. Before submission
- [ ] Run all steps on your own machine and fix any errors.
- [ ] Inspect class distribution, missing-value handling, feature names, and plots.
- [ ] Record actual metrics from your run; never invent results.
- [ ] Open MLflow and capture experiment runs/metrics.
- [ ] Run tests and capture terminal output.
- [ ] Build and run Docker; capture successful output and API response.
- [ ] Deploy to Minikube or cloud; capture pods, service, and endpoint response.
- [ ] Add genuine screenshots to `screenshots/`.
- [ ] Complete `report/REPORT_TEMPLATE.md` and export to PDF/DOCX if required.
- [ ] Add your own observations, limitations, and design rationale.
- [ ] Record a short walkthrough video showing data → training → MLflow → tests/CI → Docker/API → deployment/monitoring.
- [ ] Do not claim a public cloud URL unless you actually deployed it and verified it.
