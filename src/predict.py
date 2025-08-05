# src/predict.py

import pandas as pd
import joblib
from data_prep import clean_data, encode_categoricals

def predict_new(data_path: str, model_path: str = 'models/model.pkl', features_path: str = 'models/feature_names.pkl'):
    # Step 1: Load new data
    df = pd.read_csv(data_path)

    # Step 2: Clean and encode
    df_clean = clean_data(df)
    df_encoded = encode_categoricals(df_clean)

    # Step 3: Load model and expected features
    model = joblib.load(model_path)
    expected_features = joblib.load(features_path)

    # Step 4: Align features
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

    # Optional debug
    missing = set(expected_features) - set(df_encoded.columns)
    extra = set(df_encoded.columns) - set(expected_features)
    if missing:
        print(f"⚠️ Missing columns added with 0s: {missing}")
    if extra:
        print(f"⚠️ Unexpected columns dropped: {extra}")

    # Step 5: Predict
    predictions = model.predict(df_aligned)
    probabilities = model.predict_proba(df_aligned)[:, 1]

    # Step 6: Combine and return
    df_output = df.copy()
    df_output['Churn_Prediction'] = predictions
    df_output['Churn_Probability'] = probabilities

    return df_output

# CLI execution
if __name__ == "__main__":
    result_df = predict_new("data/raw/sample_customers.csv")
    print(result_df[['Churn_Prediction', 'Churn_Probability']].head())
