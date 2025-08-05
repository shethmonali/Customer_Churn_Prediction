

from data_prep import prepare_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_model(data_path: str, model_output: str = 'models/model.pkl'):
    X, y = prepare_features(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 0]

    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

#Save model and feature names
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')  

if __name__ == "__main__":
    train_model("/Users/monali/Documents/Customer_Churn_Prediction/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
