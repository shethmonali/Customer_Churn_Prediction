# Customer Churn Prediction (End-to-End)

## 📌 Business Problem

Customer churn is one of the most pressing challenges for subscription-based businesses. Retaining an existing customer is often 5x cheaper than acquiring a new one, and even a 5% reduction in churn can increase profits by 25–95%.

This project demonstrates how machine learning can predict at-risk customers, enabling proactive retention strategies that protect revenue and improve customer lifetime value.

## 🎯 Objective
- Predict the likelihood of customer churn using historical subscription and engagement data.  
- Provide actionable insights to marketing and customer success teams. 
- Build an end-to-end solution that could be deployed in production with monitoring and retraining.

## 📦 Project Scope
- Data cleaning and EDA
- Predictive modeling and feature engineering
- Explainability using SHAP
- Dashboards for stakeholder consumption

## 🧱 Project Structure
<img width="515" height="474" alt="Screenshot 2025-08-05 at 3 07 10 PM" src="https://github.com/user-attachments/assets/26129827-2c22-4e32-a119-3c83c5ea455e" />


## 🧑‍💼 Stakeholders
- VP of Customer Success
- Retention Marketing Team
- Data Platform Engineering

## 🧠 Model

- Algorithm: Random Forest
- AUC: ~0.87 (tunable)
- Explainability: SHAP feature importance + waterfall plots
- Trained model saved as `model.pkl`
- Feature set saved as `feature_names.pkl` to ensure consistent inference

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, Streamlit, SHAP, MLflow

## 📊 Results
- Final model AUC: 0.87
- Churn rate segments identified with 70%+ precision
- Dashboard deployed for non-technical stakeholders


