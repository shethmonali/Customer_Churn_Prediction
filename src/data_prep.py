

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric (some are empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df

def prepare_features(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_categoricals(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    return X, y
