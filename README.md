# Customer Churn Prediction Using Machine Learning

📌 Project Overview

This project focuses on predicting customer churn by analyzing historical customer data using classical supervised machine learning techniques. The objective is to identify customers who are likely to leave a service and provide insights that can support data-driven decision making.

The project is developed strictly for mid-semester evaluation and uses traditional machine learning methods only, without any GenAI or agent-based approaches.

🎯 Objectives

Analyze customer behavior data to identify churn patterns

Build a machine learning model to predict customer churn

Evaluate model performance using standard classification metrics

Deploy the solution as a simple, interactive web application

🗂️ Project Structure

customer-churn-prediction-ml/
│
├── data/
│   └── telco_churn.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── report/
    └── midsem_report.tex

📊 Dataset

Source: Telco Customer Churn Dataset
Type: Tabular data
Target Variable: Churn (Yes / No)
Key Features
    Tenure
    Monthly Charges
    Contract Type
    Payment Method
    Service Usage Information

🧠 Methodology

1. Data Preprocessing
    Handling missing values
    Encoding categorical features
    Scaling numerical features

2. Machine Learning Models
    Logistic Regression
    Decision Tree Classifier

3. Evaluation Metrics
    Accuracy
    Precision
    Recall
    F1 Score
    Confusion Matrix

    