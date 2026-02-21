# src/model_train_regression.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import get_preprocessor


def main():

    # Load dataset
    df = pd.read_csv("data/raw/loan_approval_dataset.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop ID column
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Encode loan status
    df["loan_status"] = df["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    # Keep only approved loans
    df = df[df["loan_status"] == 1].copy()

    # Log transform loan amount
    df["loan_amount"] = np.log1p(df["loan_amount"])

    # Split features and target
    X = df.drop(columns=["loan_amount", "loan_status"])
    y = df["loan_amount"]

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    preprocessor = get_preprocessor(num_cols, cat_cols)

    # Regression pipeline
    reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42
        ))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train
    reg_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = reg_pipeline.predict(X_test)

    # Reverse log transform
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)

    print("\n=== Regression Results ===")
    print("MAE :", mean_absolute_error(y_test_exp, y_pred_exp))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_exp, y_pred_exp)))
    print("R2  :", r2_score(y_test_exp, y_pred_exp))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(reg_pipeline, "models/stage_2_regressor.pkl")

    print("\nStage 2 model saved successfully.")


if __name__ == "__main__":
    main()