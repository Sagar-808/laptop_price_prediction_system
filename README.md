# Laptop Price Estimator

A complete Flask web app that estimates laptop prices using a scikit-learn Linear Regression pipeline. Clean UI, dataset-driven dropdowns, and shared preprocessing ensure consistent results.

## Features

- Dataset-driven dropdowns for all categorical fields (loaded from `laptop_data.csv`)
- Linear Regression model pipeline with ColumnTransformer (OneHotEncoder + StandardScaler)
- Shared preprocessing for training and inference to avoid train/serve skew
- Modern, responsive UI with a friendly results page and spec summary

## Project Structure

```
.
├── laptop_data.csv
├── backend/
│   ├── app.py                      # Flask app (pages + JSON APIs)
│   └── model/
│       ├── train.py                # Training entrypoint
│       ├── laptop_preprocess.py    # Feature engineering shared by train/infer
│       ├── laptop_price_model.pkl  # Generated model artifact
│       └── laptop_price_model_report.json
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── predict.html
│   │   ├── result.html
│   │   ├── about.html
│   │   └── services.html
│   └── static/
│       ├── style.css
│       └── script.js
├── requirements.txt
└── README.md
```

## Setup

Prerequisites: Python 3.10+

1. Create a virtual environment (optional) and install deps

- pip install -r requirements.txt

2. Ensure dataset is present

- Place `laptop_data.csv` at the repo root with columns:
  Company, TypeName, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight, Price

3. Train the model

- python backend/model/train.py --data laptop_data.csv --output backend/model/laptop_price_model.pkl

4. Run the app

- python backend/app.py
- Open http://localhost:5000

## Usage

- Go to /predict and select values from dropdowns (Inches is numeric)
- Submit to see the estimated price and a clear spec summary

## JSON APIs

- GET /api/options
  Returns unique values from the dataset for each dropdown field.

- POST /api/predict
  Body JSON keys mirror the dataset field names:
  { Company, TypeName, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight }
  Returns: { predicted_price: number }

## Model

- Algorithm: Custom from-scratch Linear Regression (`backend/model/linear_regression.py`)
- Split: 80/20 random
- Metrics saved to backend/model/laptop_price_model_report.json
- Currency: Model is trained on INR labels; app converts to NPR using env `INR_TO_NPR_RATE` (default 1.6).
- Engineered features include resolution (WxH, IPS, Touch), CPU brand + GHz, RAM GB, storage breakdown (SSD/HDD/Flash/Hybrid), GPU brand, Weight kg, PPI

### Evaluation (for college report)

- Train/Test Split: 80/20 random (seed=42)
- Train Accuracy (R²): 0.7947 (keys: `r2_score_train`, alias `r2_train`)
- Test Accuracy (R²): 0.7796 (keys: `r2_score_test`, alias `r2_test`)
- Error Metrics: RMSE ≈ 17,828; MAE ≈ 12,765
- Residuals: A small sample is stored in the JSON report for qualitative analysis
- Plots (saved under `backend/model/plots/`):
  - `predicted_vs_actual.png`
  - `residuals_hist.png`

Note: Confusion matrix and classification metrics (precision/recall/F1) do not apply to regression (`confusion_matrix_applicable=false`).

### Methodology Summary

1. Data preprocessing with shared feature engineering to avoid train/infer skew.
2. ColumnTransformer: OneHotEncoder (categoricals) + StandardScaler (numericals).
3. From-scratch OLS linear regression trained via NumPy least squares.
4. Metrics computed on a held-out test set; reports serialized alongside the model.

### Reproducibility

- Random seed: 42 in the train/test split.
- Command used: `python backend/model/train.py --data laptop_data.csv --output backend/model/laptop_price_model.pkl`

## Troubleshooting

- If model not found on start, run the training step above.
- If dropdowns are empty, verify laptop_data.csv exists and is readable.
- If app port is busy, edit host/port in backend/app.py.

## License

MIT
