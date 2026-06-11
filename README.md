# Loan Approval Prediction

A small machine-learning application for predicting whether a loan application is likely to be approved.

The project has three parts:

- `train_model.py` trains and saves the sklearn pipeline.
- `api/main.py` serves predictions through FastAPI.
- `dashboard/app.py` provides a Streamlit form for manual predictions.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train The Model

```powershell
python train_model.py
```

This creates:

- `model/loan_pipeline.pkl`
- `model/metadata.json`

The pipeline includes preprocessing, missing-value handling, category encoding, scaling, and logistic regression.

## Run The API

```powershell
uvicorn api.main:app --reload
```

The default API key is `loan123`. To override it:

```powershell
$env:LOAN_API_KEY = "your-secret-key"
uvicorn api.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Run The Dashboard

Start the API first, then run:

```powershell
streamlit run dashboard/app.py
```

Optional settings:

```powershell
$env:LOAN_API_URL = "http://127.0.0.1:8000"
$env:LOAN_API_KEY = "loan123"
streamlit run dashboard/app.py
```

## Example API Request

```powershell
$body = @{
  Gender = "Male"
  Married = "Yes"
  Dependents = "0"
  Education = "Graduate"
  Self_Employed = "No"
  ApplicantIncome = 5000
  CoapplicantIncome = 0
  LoanAmount = 120
  Loan_Amount_Term = 360
  Credit_History = 1.0
  Property_Area = "Urban"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -Headers @{ "X-API-KEY" = "loan123" } `
  -Body $body `
  -ContentType "application/json"
```
