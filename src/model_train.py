# model_train.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_chd_model(X_train_scaled, X_test_scaled, y_train, y_test, scaler):
    # Create and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy & report
    acc = accuracy_score(y_test, y_pred)
    print("CHD Prediction Model Trained Successfully!")
    print(f"Model Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained Logistic Regression model and Scaler
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model and Scaler saved successfully!")

    return model

