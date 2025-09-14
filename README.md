📌 Credit Risk Prediction for Loan Approval

A machine learning project to predict loan default risk based on applicant and loan-related information.
This project demonstrates an end-to-end data science pipeline including data cleaning, feature engineering, exploratory analysis, model building, hyperparameter tuning, explainability, and visualization.

🚀 Project Overview

Financial institutions often face challenges in assessing the risk of loan defaults.
This project builds a Credit Risk Prediction Model that classifies loan applications into low-risk or high-risk categories, enabling smarter decision-making.

Key highlights:

Built using Python (Pandas, NumPy, Scikit-learn, LightGBM, Optuna, SHAP)

Visual insights with Matplotlib, Seaborn, and Power BI

Achieved 97% accuracy using LightGBM with Bayesian Optimization

📂 Dataset

Source: Publicly available loan dataset (Kaggle / UCI / open-source)

Features:

Applicant income, credit score, employment length

Loan amount, loan purpose, interest rate

Repayment history & default status

⚠️ Note: Dataset is synthetic and used for learning purposes only.

🔍 Project Workflow
1. Data Wrangling & Cleaning

Removed irrelevant or low-variance columns

Handled missing values

Encoded categorical variables (e.g., loan purpose)

Normalized skewed numerical features

2. Exploratory Data Analysis (EDA)

Loan status distribution

Risk by loan purpose

Default trends across loan amount ranges

Correlation heatmap of features

3. Feature Engineering

Created loan amount bins for risk segmentation

One-hot encoding of categorical features

Class balancing with stratified splits

4. Modeling

Baseline models: Logistic Regression, Random Forest

Final model: LightGBM Classifier

Handles large tabular data efficiently

Robust to imbalance

Faster training and higher accuracy

5. Hyperparameter Tuning

Used Optuna Bayesian Optimization

Tuned learning rate, max depth, number of leaves, regularization parameters

6. Model Evaluation

Metrics: Accuracy, ROC-AUC, Confusion Matrix

Achieved 97% accuracy & 0.90+ ROC-AUC

7. Explainability (XAI)

Used SHAP values to interpret model decisions

Identified key drivers of default (e.g., loan amount, purpose, interest rate)

8. Visualization & Dashboard

Risk segmentation plots (loan purpose, loan amount)

📊 Results & Insights

Certain loan purposes had significantly higher default risks

Higher loan amounts showed increasing probability of default

Model explainability built trust and transparency in predictions

🛠️ Tech Stack

Python: Pandas, NumPy, Scikit-learn, LightGBM, Optuna, SHAP

Visualization: Matplotlib, Seaborn, Power BI

Development: Jupyter Notebook, GitHub
