"""Send random patient requests to the API so the monitoring dashboard shows traffic."""
import json
import random
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def random_patient():
    return {
        "age": random.randint(30, 77),
        "sex": random.choice([0, 1]),
        "cp": random.choice([1, 2, 3, 4]),
        "trestbps": random.randint(95, 190),
        "chol": random.randint(130, 400),
        "fbs": random.choice([0, 1]),
        "restecg": random.choice([0, 1, 2]),
        "thalach": random.randint(90, 200),
        "exang": random.choice([0, 1]),
        "oldpeak": round(random.uniform(0, 5), 1),
        "slope": random.choice([1, 2, 3]),
        "ca": random.choice([0, 1, 2, 3]),
        "thal": random.choice([3, 6, 7]),
    }


def post(path, payload):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def main():
    counts = {}
    errors = 0
    for i in range(N):
        try:
            if i % 25 == 0:
                post("/predict", {"age": "abc"})  # invalid on purpose -> 422
            result = post("/predict", random_patient())
            counts[result["label"]] = counts.get(result["label"], 0) + 1
        except urllib.error.HTTPError:
            errors += 1
        if i % 10 == 0:
            urllib.request.urlopen(URL + "/health", timeout=5).read()
        time.sleep(0.1)
    print(f"sent {N} predictions to {URL}: {counts}, errors: {errors}")


if __name__ == "__main__":
    main()
