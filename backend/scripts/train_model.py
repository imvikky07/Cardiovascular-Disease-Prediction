"""
Train and save the CVD Risk Logistic Regression model.
Run from the /backend directory: python scripts/train_model.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'ml')


def train():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset shape: {df.shape}")

    df = df.dropna()

    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    feature_cols = ['age', 'gender', 'cholesterol', 'blood_pressure', 'glucose',
                    'smoking', 'alcohol', 'bmi', 'physical_activity']
    X = df[feature_cols]
    y = df['cvd_risk']

    logger.info(f"Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    logger.info(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_names.joblib'))

    logger.info("All model artifacts saved to app/ml/")


if __name__ == "__main__":
    train()
