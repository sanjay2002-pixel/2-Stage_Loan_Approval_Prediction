#  2-Stage Loan Approval & Valuation System

A production-ready, end-to-end Machine Learning application that simulates how financial institutions evaluate loan applications using a two-stage modeling approach.

🔗 **Live Demo:**https://2-stageloanapprovalprediction-h39vsueaskfsz6jqwhbfmy.streamlit.app/

##  Problem Statement

Financial institutions must evaluate loan applications efficiently while minimizing financial risk.

This project implements a **two-stage machine learning system**:

- **Stage 1 – Classification:**  
  Predict whether a loan application should be **Approved** or **Rejected**.

- **Stage 2 – Regression:**  
  If approved, predict the **optimal loan amount** using a valuation model.



##  Dataset Information

- 📄 Dataset: Loan Approval Dataset
- 📏 Size: 4269 rows × 13 columns
- 🧾 Features include:
  - Number of Dependents
  - Education
  - Self Employed Status
  - Annual Income
  - CIBIL Score
  - Asset Values
  - Loan Term
  - Loan Amount
  - Loan Status (Target)




##  Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Git & GitHub



##  Machine Learning Approach

###  Stage 1 – Classification

- Model: Random Forest Classifier
- Evaluation Metrics:
  - Accuracy: ~98%
  - Recall: ~98%
  - F1 Score: High balanced performance

###  Stage 2 – Regression

- Model: Random Forest Regressor
- Log transformation applied to stabilize variance
- Evaluation Metrics:
  - R² Score: ~0.87
  - MAE: ~2.5M (approx)
  - RMSE: ~3.3M



##  Key Features

✔ Modular preprocessing pipeline using `ColumnTransformer`  
✔ Separate classification and regression models  
✔ Production-ready architecture (`src/` structure)  
✔ Model serialization using Joblib  
✔ Streamlit web interface  
✔ Cloud deployment  



##  How to Run Locally

```bash
# Clone repository
git clone https://github.com/sanjay2002-pixel/2-stage_loan_approval_prediction.git

# Navigate into project
cd 2-stage_loan_approval_prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

