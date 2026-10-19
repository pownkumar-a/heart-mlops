import json
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/model.joblib")

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(records, model=None):
    model = model or load_model()
    df = pd.DataFrame(records).astype(float)
    proba = model.predict_proba(df)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        results.append({
            "prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "confidence": round(float(max(p, 1 - p)), 3),
            "disease_probability": round(float(p), 3),
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        patients = json.loads(Path(sys.argv[1]).read_text())
    else:
        patients = [SAMPLE_PATIENT]
    print(json.dumps(predict(patients), indent=2))
