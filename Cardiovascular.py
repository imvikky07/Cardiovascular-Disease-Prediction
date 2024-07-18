# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:54:50 2024

@author: Vivek
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('cardio_train.csv', sep=';')

# Step 2: Data Preprocessing
# Handle missing values
df = df.dropna()

# Feature scaling and encoding
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Train various machine learning models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42)
}

# Dictionary to store the results
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Step 5: Print the results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Step 6: Example prediction
example_patient = np.array([[50, 1, 160, 70, 120, 80, 1, 0, 0, 0, 1]])
example_patient_scaled = scaler.transform(example_patient)
best_model = models['Random Forest']  # Assuming Random Forest performed best
prediction = best_model.predict(example_patient_scaled)
print(f"Prediction for example patient: {'Cardiovascular Disease' if prediction[0] == 1 else 'No Cardiovascular Disease'}")
