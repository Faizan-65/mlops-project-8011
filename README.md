# MLOps Assignment 1 — 25L-8011

## Structure
```
data/    raw dataset — dataset.csv
src/     training script — train_25L-8011.py
model/   trained artifact — model.joblib
```

## Setup & run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_25L-8011.py
```

Trains a RandomForest regressor on `data/dataset.csv` predicting `price`,
prints test R2 / MAE, and writes `model/model.joblib`.
