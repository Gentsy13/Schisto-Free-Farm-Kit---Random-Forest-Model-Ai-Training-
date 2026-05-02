# Schisto-Free-Farm-Kit---Random-Forest-Model-Ai-Training-
Training for AI model using publicly available CSV datasets.

## Setup

Install dependencies in the workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib scikit-learn joblib
```

Place `water_sensor_data.csv` in the project root.

## Train The Model

```powershell
.\.venv\Scripts\python.exe .\train.py
```

Outputs created after training:
- `water_level_model.joblib` (trained model + feature order)
- `robustness_plot.png` (actual vs predicted plot)

To show the chart window interactively:

```powershell
$env:SHOW_PLOT='1'; .\.venv\Scripts\python.exe .\train.py
```

## Predict A Single Sample

Use the saved model to predict from one sensor reading:

```powershell
.\.venv\Scripts\python.exe .\predict.py --values 151 4800 148.5 1.024 0.152 2.044 -0.003 0.970 -0.145 -0.465 0.503 -0.267
```
