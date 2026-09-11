# MLOps Assignment 1 — 25L-8011

## Structure
```
data/    dataset.csv (raw) + dataset_clean.csv (preprocessed, one-hot encoded)
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

Preprocesses `data/dataset.csv` into `data/dataset_clean.csv` (nulls and duplicates
dropped, categorical columns one-hot encoded), trains a RandomForest regressor on that
clean file to predict `price`, prints test R2 / MAE, and writes `model/model.joblib`.
