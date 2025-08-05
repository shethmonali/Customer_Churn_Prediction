
import streamlit as st
import pandas as pd
import joblib
from data_prep import clean_data, encode_categoricals

st.title("🔁 Customer Churn Predictor")

uploaded_file = st.file_uploader("Upload a CSV of customer data", type=["csv"])

if uploaded_file:
    # Step 1: Load and process data
    df = pd.read_csv(uploaded_file)
    df_clean = clean_data(df)
    df_encoded = encode_categoricals(df_clean)

    # Step 2: Load model + expected features
    model = joblib.load("/Users/monali/Documents/Customer_Churn_Prediction/models/model.pkl")
    expected_features = joblib.load("/Users/monali/Documents/Customer_Churn_Prediction/models/feature_names.pkl")

    # Step 3: Align features
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

    # Step 4: Predict
    predictions = model.predict(df_aligned)
    probabilities = model.predict_proba(df_aligned)[:, 0]

    # Step 5: Show predictions
    df['Churn_Prediction'] = predictions
    df['Churn_Probability'] = probabilities

    st.write("### Predictions")
    st.dataframe(df[['Churn_Prediction', 'Churn_Probability'] + [col for col in df.columns if col not in ['Churn_Prediction', 'Churn_Probability']]])

    st.write("### Churn Prediction Breakdown")
    st.bar_chart(df['Churn_Prediction'].value_counts(normalize=True))


