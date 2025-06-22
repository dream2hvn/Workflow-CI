import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="diabetes_clean.csv")
args = parser.parse_args()

# Autolog
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Inisialisasi model LogisticRegression
model = LogisticRegression(max_iter=200, random_state=42)

# Mulai run MLflow untuk training dan logging
with mlflow.start_run():
    model.fit(X_train, y_train)
    
    # Log model secara eksplisit (meskipun autolog sudah aktif)
    mlflow.sklearn.log_model(model, "model")
    
    # Hitung akurasi pada data test
    accuracy = model.score(X_test, y_test)
    
    # Log akurasi ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
