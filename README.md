# Loan Approval Prediction

A Streamlit machine-learning app that predicts whether a loan application is likely to be approved.

The app runs locally with Python. It does not need FastAPI, Uvicorn, or a separate backend server.

## Project Files

- `train_model.py` trains the model pipeline.
- `dashboard/app.py` runs the Streamlit app.
- `model/loan_pipeline.pkl` is the trained model pipeline.
- `model/metadata.json` stores model metrics and feature details.
- `train.csv` is the training dataset.
- `test.csv` is the test dataset.
- `loan(lr).ipynb` is the original notebook.

## Setup

From the project folder:

```powershell
cd C:\Users\user\Downloads\loan-approval
```

Install the required packages:

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Train The Model

Run this once before starting the app:

```powershell
py -3.11 train_model.py
```

This creates or updates:

- `model/loan_pipeline.pkl`
- `model/metadata.json`

The pipeline handles missing values, categorical encoding, scaling, and logistic regression.

## Run The App

Start Streamlit:

```powershell
py -3.11 -m streamlit run dashboard/app.py
```

Then open:

[http://localhost:8501](http://localhost:8501)

## Prediction Inputs

The app asks for:

- Applicant information
- Income information
- Loan amount and loan term
- Credit history
- Property area

It returns:

- Approved or rejected prediction
- Approval probability

## Notes

If the app says the model is missing, run:

```powershell
py -3.11 train_model.py
```

Then restart Streamlit.
