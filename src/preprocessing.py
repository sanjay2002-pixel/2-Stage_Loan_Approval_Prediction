# src/preprocessing.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_preprocessor(num_cols, cat_cols):
    """
    Returns a ColumnTransformer preprocessing pipeline.

    Parameters:
    ----------
    num_cols : list
        List of numerical column names.
    cat_cols : list
        List of categorical column names.

    Returns:
    -------
    ColumnTransformer
        Preprocessing pipeline.
    """

    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor