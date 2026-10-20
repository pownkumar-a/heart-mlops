# Heart Disease Risk Prediction - End-to-End MLOps Pipeline

**BITS Pilani WILP | AIMLCZG523 Machine Learning Operations | Assignment 1**
**Student:** PownKumar A
**Student ID:** 2025AE05941

 **Report:** [Final report (PDF)](reports/2025AE05941_MLOps_Assignment1_Report.pdf)
 **Demo video:** [Watch the pipeline demo (Google Drive)](https://drive.google.com/drive/folders/1LG2Mi2sufPHRMZ40bPn73uhJcZ_HcPEJ)

A machine-learning classifier that predicts the presence of heart disease from 13 clinical
features (UCI Heart Disease dataset, Cleveland, 303 patients). The model is served as a REST API
with experiment tracking, automated tests, CI/CD, Docker, Kubernetes and monitoring.

## Pipeline overview
```
UCI dataset -> EDA -> Training (Logistic Regression vs Random Forest, GridSearchCV)
  -> MLflow tracking -> models/model.joblib -> pytest + flake8 -> GitHub Actions CI
  -> Docker image (FastAPI) -> Kubernetes (minikube, 2 replicas, LoadBalancer)
  -> Prometheus + Grafana monitoring
```

## Results
5-fold stratified cross-validation on the 80% training split (242 rows); 20% hold-out test set (61 rows).

| Model | CV ROC-AUC | CV Accuracy | CV F1 | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (C=0.3)** - selected | **0.909 ± 0.018** | 0.847 | 0.824 | 0.885 | 0.929 | 0.959 |
| Random Forest (300 trees, depth 5, leaf 5) | 0.901 ± 0.025 | 0.835 | 0.813 | 0.902 | 0.929 | 0.960 |

The model is selected by **CV ROC-AUC only**. The test set is used for reporting, never for selection.
Logistic Regression was chosen: higher CV ROC-AUC, lower variance, and it is simpler and interpretable.

## Repository structure
| Path | Contents |
|---|---|
| `src/download_data.py` | Downloads the UCI dataset -> `data/raw/heart.csv` |
| `src/eda.py`, `notebooks/01_eda.ipynb` | EDA plots -> `reports/figures/` |
| `src/train.py`, `notebooks/02_training.ipynb` | Preprocessing pipeline, GridSearchCV, MLflow logging |
| `src/predict.py`, `notebooks/03_inference.ipynb` | Load the saved model and predict |
| `models/` | `model.joblib` (full sklearn pipeline) and `metrics.json` |
| `tests/`, `notebooks/04_tests_ci.ipynb` | 9 pytest unit tests (data, pipeline, prediction) |
| `.github/workflows/ci.yml` | CI: flake8 -> pytest -> train; uploads test results and model artifacts |
| `app/main.py`, `Dockerfile`, `notebooks/05_api_docker.ipynb` | FastAPI service with request logging and Prometheus metrics |
| `k8s/` | Deployment (2 replicas, readiness/liveness probes) and LoadBalancer Service |
| `monitoring/`, `notebooks/06_monitoring.ipynb` | Prometheus config and Grafana dashboard |
| `src/generate_traffic.py` | Sends test traffic to the API |
| `screenshots/` | Evidence for every task |

## Setup and run
```bash
git clone https://github.com/pownkumar-a/heart-mlops.git
cd heart-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py      # 1. download data
python src/eda.py                # 2. EDA figures
python src/train.py              # 3. train models + MLflow logging
mlflow ui --port 5000            # 4. view experiments at http://localhost:5000
flake8 src tests && pytest -v    # 5. lint + unit tests
```

## Docker
```bash
docker build -t heart-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0
curl http://localhost:8000/health
```

## Kubernetes (minikube)
```bash
minikube start
minikube image load heart-api:1.0
kubectl apply -f k8s/
kubectl get deployments,pods,svc
minikube service heart-api --url      # prints the API URL, e.g. http://192.168.49.2:30080
```

## API access
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Welcome message |
| `/health` | GET | Health check (used by Kubernetes probes) |
| `/predict` | POST | Predict heart disease for one patient |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive Swagger UI |

**Example request**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
**Example response**
```json
{"prediction": 0, "label": "no heart disease", "confidence": 0.714, "disease_probability": 0.286}
```
Invalid input (e.g. a missing field or wrong type) returns HTTP 422.

## Monitoring
```bash
docker network create mlops-net
docker run -d --name heart-api --network mlops-net -p 8000:8000 heart-api:1.0
docker create --name prometheus --network mlops-net -p 9090:9090 prom/prometheus:v2.53.0
docker cp monitoring/prometheus.yml prometheus:/etc/prometheus/prometheus.yml
docker start prometheus
docker create --name grafana --network mlops-net -p 3000:3000 -e GF_SECURITY_ADMIN_PASSWORD=admin grafana/grafana:11.1.0
docker cp monitoring/grafana/provisioning/. grafana:/etc/grafana/provisioning/
docker start grafana
python src/generate_traffic.py http://localhost:8000 300
```
- Prometheus: http://localhost:9090/targets
- Grafana: http://localhost:3000 (admin/admin) -> Dashboards -> MLOps -> Heart API Monitoring
- Metrics: `api_requests_total`, `api_request_latency_seconds`, `predictions_total`
- Logs: every request is logged as `METHOD path -> status (ms)` (`docker logs heart-api` / `kubectl logs -l app=heart-api`)

## Dataset
Detrano, R. et al. (1989). UCI Heart Disease Dataset, Cleveland Clinic Foundation.
https://archive.ics.uci.edu/dataset/45/heart+disease

