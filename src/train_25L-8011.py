"""Train a RandomForest on data/dataset.csv and save it to model/."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.csv"
MODEL = ROOT / "model" / "model.joblib"


# ponytail: last column is the target, swap for a --target flag if that stops holding
def main():
    if not DATA.exists():
        sys.exit(f"No dataset at {DATA}")

    df = pd.read_csv(DATA)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    print(f"Loaded {DATA} -> {df.shape[0]} rows, {df.shape[1] - 1} features")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    print(f"Test accuracy: {accuracy_score(y_te, clf.predict(X_te)):.3f}")

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()
