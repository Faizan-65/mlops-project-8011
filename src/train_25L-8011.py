"""Train a RandomForest regressor on data/Housing.csv and save it to model/."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.csv"
MODEL = ROOT / "model" / "model.joblib"
TARGET = "price"


def main():
    if not DATA.exists():
        sys.exit(f"No dataset at {DATA}")

    df = pd.read_csv(DATA).dropna()
    if df.empty:
        sys.exit(f"{DATA} has a header but no rows")
    print(f"Loaded {DATA} -> {df.shape[0]} rows, {df.shape[1] - 1} features")

    X = pd.get_dummies(df.drop(columns=TARGET))
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Test R2: {r2_score(y_te, pred):.3f}  MAE: {mean_absolute_error(y_te, pred):,.0f}")

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()
