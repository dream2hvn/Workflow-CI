import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Aktifkan autologging dari MLflow
mlflow.sklearn.autolog()

# Jangan set tracking URI, biarkan default local storage MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Hapus atau komentari baris ini

# Set experiment seperti biasa
mlflow.set_experiment("iris_experiment")

# Load dataset hasil preprocessing iris
df = pd.read_csv("iris_preprocessed.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["species"])
y = df["species"]

# Split data menjadi train dan test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
