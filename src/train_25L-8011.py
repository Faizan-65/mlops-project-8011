"""Preprocess data/dataset.csv into a clean one-hot encoded CSV, then train on it."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "dataset.csv"
CLEAN = ROOT / "data" / "dataset_clean.csv"
MODEL = ROOT / "model" / "model.joblib"
TARGET = "price"

# Hyperparameters — bump these between runs to compare experiments
N_ESTIMATORS = 100
MAX_DEPTH = None
RANDOM_STATE = 42


def preprocess():
    """Raw CSV -> fully numeric CSV with the categorical columns one-hot encoded."""
    if not RAW.exists():
        sys.exit(f"No dataset at {RAW}")

    df = pd.read_csv(RAW)
    before = len(df)
    df = df.dropna().drop_duplicates()
    print(f"Loaded {RAW.name}: {before} rows -> {len(df)} after dropping nulls/duplicates")
    if df.empty:
        sys.exit(f"{RAW} has a header but no usable rows")

    cats = list(df.drop(columns=TARGET).select_dtypes(exclude="number").columns)
    # ponytail: get_dummies is the one-hot; OneHotEncoder earns its keep once
    # unseen categories have to be handled at inference time
    clean = pd.get_dummies(df, columns=cats, dtype=int)
    print(f"One-hot encoded {len(cats)} categorical columns: {', '.join(cats)}")

    clean.to_csv(CLEAN, index=False)
    print(f"Wrote {CLEAN.name}: {clean.shape[0]} rows, {clean.shape[1] - 1} features")
    return clean


def train(clean):
    X, y = clean.drop(columns=TARGET), clean[TARGET]
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)  # min-max scaling
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print(f"Training RandomForest: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    ).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Test R2: {r2_score(y_te, pred):.3f}  MAE: {mean_absolute_error(y_te, pred):,.0f}")

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    train(preprocess())
