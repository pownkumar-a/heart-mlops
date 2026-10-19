import json
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  # noqa: E402

DATA_PATH = Path("data/raw/heart.csv")
MODEL_DIR = Path("models")
PLOT_DIR = Path("reports/figures")
SEED = 42
EXPERIMENT = "heart-disease"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = (df["num"] > 0).astype(int)  # 1 = heart disease
    X = df[FEATURES]
    return X, y


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", binary, BINARY),
        ("cat", categorical, CATEGORICAL),
    ])


def candidate_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 5]},
        ),
    }


def test_scores(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def save_plots(name, model, X_test, y_test):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(PLOT_DIR / f"cm_{name}.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title(f"ROC curve - {name}")
    paths.append(PLOT_DIR / f"roc_{name}.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)

    results = {}
    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])

        with mlflow.start_run(run_name=name) as run:
            # 1. Tune hyper-parameters with 5-fold CV (score = ROC-AUC)
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc")
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Log every grid candidate as a child run
            for i, params in enumerate(search.cv_results_["params"]):
                with mlflow.start_run(run_name=f"{name}-{i}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_roc_auc", search.cv_results_["mean_test_score"][i])

            # 2. Cross-validated metrics 
            cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=METRICS)
            cv_mean = {m: cv_res[f"test_{m}"].mean() for m in METRICS}
            cv_std = {m: cv_res[f"test_{m}"].std() for m in METRICS}

            # 3. Final check on the untouched test set
            test = test_scores(best, X_test, y_test)

            # 4. Log parameters, metrics, plots and the model to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({f"cv_{m}_mean": v for m, v in cv_mean.items()})
            mlflow.log_metrics({f"cv_{m}_std": v for m, v in cv_std.items()})
            mlflow.log_metrics({f"test_{m}": v for m, v in test.items()})
            for path in save_plots(name, best, X_test, y_test):
                mlflow.log_artifact(str(path), artifact_path="plots")
            mlflow.sklearn.log_model(best, artifact_path="model",
                                     input_example=X_test.head(2))

        results[name] = {"best_params": search.best_params_, "cv_mean": cv_mean,
                         "cv_std": cv_std, "test": test, "pipeline": best,
                         "mlflow_run_id": run.info.run_id}
        print(f"\n=== {name} ===")
        print("best params:", search.best_params_)
        for m in METRICS:
            print(f"  {m:9s}  CV {cv_mean[m]:.3f} ± {cv_std[m]:.3f}   test {test[m]:.3f}")

    
    winner = max(results, key=lambda n: results[n]["cv_mean"]["roc_auc"])
    print(f"\nSelected model: {winner}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(results[winner]["pipeline"], MODEL_DIR / "model.joblib")
    summary = {n: {k: v for k, v in r.items() if k != "pipeline"} for n, r in results.items()}
    summary["selected_model"] = winner
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Saved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
