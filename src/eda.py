"""Exploratory data analysis: prints a summary and saves charts to reports/figures/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/heart.csv")
FIG_DIR = Path("reports/figures")

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["num"] > 0).astype(int)  # 1 = heart disease, 0 = none

    print("Summary statistics:")
    print(df.describe().T.round(1))
    print("\nClass balance (0 = no disease, 1 = disease):")
    print(df["target"].value_counts())

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="target", ax=ax)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    save(fig, "01_class_balance.png")

    # 2. Distribution of each numeric feature, split by diagnosis
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(data=df, x=col, hue="target", kde=True,
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    save(fig, "02_numeric_histograms.png")

    # 3. Disease rate for each value of the categorical features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, CATEGORICAL):
        sns.barplot(data=df, x=col, y="target", errorbar=None, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("disease rate")
        ax.set_title(col)
    save(fig, "03_categorical_disease_rate.png")

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.drop(columns="num").corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "04_correlation_heatmap.png")

    # 5. Box plots to spot outliers
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x="target", y=col, ax=ax)
        ax.set_title(col)
    save(fig, "05_boxplots.png")


if __name__ == "__main__":
    main()
