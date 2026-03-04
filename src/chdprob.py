import joblib
import pandas as pd

# Load models from src/
log_reg = joblib.load("logistic_model.pkl")  # saved in src/
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("../data/raw/projectdataset.csv")

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = log_reg.predict(X_scaled)
y_prob = log_reg.predict_proba(X_scaled)[:, 1]

results_df = pd.DataFrame({
    'Actual_CHD': y.values,
    'Predicted_CHD': y_pred,
    'CHD_Probability': y_prob
})

print("Predicted CHD Probability for first 10 samples:")
print(results_df.head(10))