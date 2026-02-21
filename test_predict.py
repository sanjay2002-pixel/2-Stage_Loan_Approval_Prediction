from src.predict import LoanPredictor
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Take one sample row
sample = df.drop(columns=["loan_status"]).iloc[0].to_dict()

# Initialize predictor
predictor = LoanPredictor()

# Predict
result = predictor.predict(sample)

print(result)