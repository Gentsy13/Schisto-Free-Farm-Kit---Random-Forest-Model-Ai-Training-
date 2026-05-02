# Schisto-Free Farm Kit – Random Forest AI Model

A Python + scikit-learn project that trains a **Random Forest classifier** to predict
schistosomiasis risk on farms based on environmental, behavioural, and infrastructure features.

---

## Project Structure

```
.
├── data/
│   ├── generate_sample_data.py   # Script to generate a synthetic training CSV
│   └── schisto_risk.csv          # 1 000-row sample dataset (auto-generated)
├── models/
│   ├── rf_model.joblib           # Saved trained model (created after training)
│   └── plots/
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── src/
│   ├── preprocess.py             # Data loading & preprocessing pipeline
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Evaluation helpers (metrics, plots)
│   └── predict.py                # Inference / batch prediction script
├── tests/
│   └── test_pipeline.py          # Pytest unit tests (15 tests)
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .          # installs the src package in editable mode
```

### 2. Generate the sample dataset

```bash
python data/generate_sample_data.py
```

This creates `data/schisto_risk.csv` with 1 000 labelled rows.
Replace it (or add to it) with your own real-world CSV that follows the same column schema.

### 3. Train the model

```bash
python -m src.train
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/schisto_risk.csv` | Path to training CSV |
| `--output` | `models/rf_model.joblib` | Where to save the trained model |
| `--n_estimators` | `200` | Number of trees in the forest |
| `--max_depth` | `None` | Max depth per tree (None = unlimited) |
| `--test_size` | `0.2` | Fraction of data reserved for testing |
| `--random_state` | `42` | Random seed for reproducibility |
| `--tune` | *(flag)* | Run GridSearchCV hyperparameter search |

Example with custom options:

```bash
python -m src.train --n_estimators 300 --max_depth 20 --tune
```

### 4. Run predictions on new data

```bash
python -m src.predict --model models/rf_model.joblib --input data/new_samples.csv
```

Add `--output predictions.csv` to save results to disk.

---

## Dataset Features

| Column | Type | Description |
|--------|------|-------------|
| `proximity_to_water_m` | numeric | Distance to nearest water body (metres) |
| `water_contact_hrs_per_day` | numeric | Hours of daily water contact |
| `irrigation_type` | int (0-2) | 0 = drip, 1 = flood, 2 = sprinkler |
| `soil_moisture_pct` | numeric | Soil moisture percentage |
| `temperature_c` | numeric | Ambient temperature (°C) |
| `rainfall_mm` | numeric | Monthly rainfall (mm) |
| `snail_habitat_index` | float (0-1) | Suitability for intermediate snail hosts |
| `sanitation_score` | int (0-10) | Higher = better sanitation facilities |
| `age_group` | int (0-2) | 0 = child, 1 = adult, 2 = elder |
| `ppe_usage` | int (0-1) | 0 = no PPE, 1 = PPE used |
| **`schisto_risk`** *(target)* | int (0-1) | **0 = Low Risk, 1 = High Risk** |

---

## Model Performance (sample data)

| Metric | Score |
|--------|-------|
| Accuracy | **0.88** |
| ROC-AUC | **0.97** |
| High-Risk Recall | **0.93** |
| High-Risk Precision | **0.87** |

Top predictors identified by the model:

1. `snail_habitat_index` (0.233)
2. `water_contact_hrs_per_day` (0.210)
3. `proximity_to_water_m` (0.146)

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 15 unit tests cover preprocessing, training, and prediction.

---

## Bring Your Own Data

1. Prepare a CSV with the columns listed in the **Dataset Features** table above.
2. Point `--data` to your CSV when training.
3. For purely unlabelled inference (no `schisto_risk` column), use `predict.py` directly.
