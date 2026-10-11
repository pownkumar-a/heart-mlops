"""to download the dataset"""
from pathlib import Path

import pandas as pd

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

OUT_PATH = Path("data/raw/heart.csv")


def main():
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved to", OUT_PATH)
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget 'num' counts:")
    print(df["num"].value_counts().sort_index())


if __name__ == "__main__":
    main()