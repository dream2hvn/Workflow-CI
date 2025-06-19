import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Aktifkan autologging dari MLflow
mlflow.sklearn.autolog()

# Atur URI tracking dan eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000")
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

# Inisialisasi model RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

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
