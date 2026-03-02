# Customer Churn Prediction using Machine Learning

## Project Overview

This project focuses on the design and implementation of a machine learning–based customer churn prediction system using historical customer behavior data. The objective is to identify customers who are likely to discontinue services and provide actionable insights that support data-driven retention strategies.
The system uses classical supervised learning techniques and provides an interactive web interface for real-time churn prediction and analysis.

## Problem Motivation

Customer churn is a major challenge in subscription-based industries such as telecommunications. Customers may leave due to multiple factors including contract type, service usage, billing patterns, and tenure.

Accurately predicting churn allows organizations to:
- Identify high-risk customers early
- Reduce revenue loss
- Design targeted retention strategies

This project models churn as a **binary classification problem** using structured customer data.


## Project Structure

## Technology Stack

| Component                | Technology                          |
| ------------------------ | ----------------------------------- |
|   Programming Language** | Python                              |
|   ML Framework**         | Scikit-learn                        |
|   Model Used**           | Logistic Regression / Random Forest |
|   Data Processing**      | Pandas, NumPy                       |
|   Web Interface**        | Streamlit                           |
|   Model Persistence**    | Joblib                              |
|   Deployment Platform**  | Streamlit Cloud                     |

## System Architecture

1. **Data Ingestion**
   - CSV-based dataset upload
   - Schema validation

2. **Preprocessing Pipeline**
   - Encoding of categorical features
   - Scaling of numerical features
   - Unified pipeline for training and inference

3. **Model Training**
   - Train–test data split
   - Supervised classification model
   - Hyperparameter-controlled learning

4. **Prediction & Visualization**
   - Real-time churn probability prediction
   - Binary churn classification
   - Interactive data filtering and visualization

#### Milestones & Deliverables

### Milestone 1: ML-Based Churn Prediction (Mid-Sem)

**Objective:**  
Develop an end-to-end churn prediction pipeline using classical machine learning techniques.

**Deliverables:**
- Problem formulation and dataset analysis  
- Feature engineering and preprocessing pipeline  
- Trained churn prediction model  
- Model evaluation using standard metrics  
- Streamlit-based interactive web application  
- Well-structured GitHub repository  

### Milestone 2: Agentic AI–Based Retention Assistant (End-Sem)

**Objective:**  
Extend the churn prediction system into an **agentic retention assistant** that autonomously reasons over churn risk, customer segments, and historical best practices to generate **structured, actionable retention recommendations**.

**Scope & Capabilities:**
- Analyze predicted churn probabilities and customer profiles
- Identify high-risk customer segments using rule-based reasoning
- Retrieve predefined retention strategies and best-practice policies
- Generate structured recommendations (offers, service actions, follow-ups)
- Provide explainable reasoning for each suggested retention action

**Key Outcomes:**
- Transition from prediction-only analytics to decision-support intelligence
- Demonstrate autonomous reasoning and strategy selection
- Enable scenario-based retention planning for business users

**Note:**  
This milestone builds upon the classical ML foundation developed in Milestone 1 and introduces agentic reasoning for decision support as part of the end-semester extension.

## Dataset Description

- **Dataset:** Telco Customer Churn Dataset  
- **Records:** ~7,000 customers  
- **Features:** Demographic details, service usage, billing information  
- **Target Variable:** Churn (Yes / No)

Non-predictive identifiers such as customer ID are removed during training to ensure model integrity.

## Web Application Features
  
- Interactive sidebar filters for customer segmentation  
- Dataset preview and summary statistics  
- Real-time churn probability prediction  
- Binary churn classification  
- Downloadable prediction results  

The application is designed to be intuitive, stable, and suitable for live demonstration.

## 👥 Team Members

* Rudraksh Rathod - 2401010396
* Kaustubh Hiwanj - 2401010217
* Bhargav Patil - 2401020092
