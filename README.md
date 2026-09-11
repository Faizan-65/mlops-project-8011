# MLOps Assignment 1 — 25L-8011

## Structure
```
data/    raw dataset (dataset.csv, git-ignored)
src/     training script — train_25L-8011.py
model/   trained artifact (model.joblib, git-ignored)
```

## Setup & run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_25L-8011.py
```

Put your dataset at `data/dataset.csv` first — the **last column is treated as the target**,
every other column as a feature. The script prints the test accuracy and writes
`model/model.joblib`.
