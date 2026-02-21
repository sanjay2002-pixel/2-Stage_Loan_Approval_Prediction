# src/model_train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score

from src.preprocessing import get_preprocessor


def main():

    # Load dataset
    df = pd.read_csv("data/raw/loan_approval_dataset.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop ID column if exists
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Encode target
    df["loan_status"] = df["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    # Split features and target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = get_preprocessor(num_cols, cat_cols)

    # Full classification pipeline
    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_split=5,
            random_state=42
        ))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Train model
    clf_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = clf_pipeline.predict(X_test)

    print("\n=== Classification Results ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf_pipeline, "models/stage_1_classifier.pkl")

    print("\nStage 1 model saved successfully.")


if __name__ == "__main__":
    main()