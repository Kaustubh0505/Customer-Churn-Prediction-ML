# Model Evaluation Report

**Project:** Customer Churn Prediction using Random Forest Classifier  
**Capstone:** Milestone 1  
**Dataset:** Telco Customer Churn Dataset (Kaggle)  

---

## 1. Executive Summary

This report explains how well the machine learning model performs in predicting customer churn. Customer churn means customers leaving a telecom service. Predicting churn is important because keeping existing customers costs less than finding new ones.

The model uses customer details such as services used, payment information, and contract type to predict whether a customer will **churn (1)** or **not churn (0)**.

### Key Results

- **Testing Accuracy:** 80.7%
- **Weighted F1-Score:** 0.80
- **Precision for Churn Customers:** 0.68
- **Recall for Churn Customers:** 0.52

Overall, the model performs well and gives reliable results, even though the dataset has fewer churn customers.

---

## 2. Model Details

- **Algorithm Used:** Random Forest Classifier  
- **Type:** Supervised Learning (Binary Classification)  
- **Library:** Scikit-Learn  
- **Target Column:** `Churn`

---

## 3. Dataset Overview

- **Total Customers:** 7,043  
- **Total Features:** 21  
- **Class Distribution:**
  - Customers who did not churn: ~73%
  - Customers who churned: ~27%

### Data Preprocessing Steps

- Removed the `customerID` column  
- Converted categorical data into numerical form using label encoding  
- Split the data into **80% training** and **20% testing**

---

## 4. Model Architecture

Random Forest was chosen because it combines multiple decision trees, which helps reduce overfitting and improve accuracy.

### Hyperparameters Used

- `n_estimators`: 200  
- `max_depth`: 11  
- `min_samples_split`: 5  
- `min_samples_leaf`: 2  

---

## 5. Model Performance

The model was evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|---------|---------|
| No Churn (0) | 0.84 | 0.91 | 0.88 | 1036 |
| Churn (1) | 0.68 | 0.52 | 0.59 | 373 |

### Overall Results

- **Accuracy:** 89%  
- **Macro Average F1-Score:** 0.73  
- **Weighted Average F1-Score:** 0.80  

---

## 6. Business Importance

- Helps the company identify customers who might leave  
- Allows early action to retain customers  
- Reduces loss of revenue  
- Supports better business decisions

---

## 7. Conclusion

The Random Forest model gives strong and reliable results with over **89% accuracy**. It meets the goals of Capstone Milestone 1 and provides a solid base for future improvements in customer retention systems.
