# src/predict.py

import joblib
import pandas as pd
import numpy as np


class LoanPredictor:

    def __init__(self):
        """
        Load both trained models.
        """
        self.classifier = joblib.load("models/stage_1_classifier.pkl")
        self.regressor = joblib.load("models/stage_2_regressor.pkl")

    def predict(self, input_data: dict):
        """
        input_data: dictionary containing loan applicant details
        """

        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])

        # Clean column names
        df.columns = df.columns.str.strip()

        # Clean string values
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # -------- Stage 1: Classification --------
        approval = self.classifier.predict(df)[0]

        if approval == 1:
            # -------- Stage 2: Regression --------
            amount_log = self.regressor.predict(df)[0]
            amount = np.expm1(amount_log)  # reverse log transform

            return {
                "Loan Status": "Approved",
                "Predicted Loan Amount": round(float(amount), 2)
            }

        else:
            return {
                "Loan Status": "Rejected",
                "Predicted Loan Amount": None
            }