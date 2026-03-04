# app.py
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd

# Setup paths
ROOT = Path(__file__).resolve().parent.parent  # Project root
SRC_PATH = ROOT / "src"
sys.path.append(str(SRC_PATH))

from health_score_kmeans import predict_health_score_for_new
from recommend_insurance import recommend_insurance

app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_key"

# Model paths
MODEL_PATH = ROOT / "logistic_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"


def load_models():
    """Load logistic model and scaler safely"""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run training first.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


try:
    model, scaler = load_models()
    load_error = None
except Exception as e:
    load_error = str(e)
    model = None
    scaler = None


def parse_float(value, name, min_val=None, max_val=None, default=None):
    """Parse and validate numeric input values"""
    try:
        v = float(value)
    except Exception:
        raise ValueError(f"Invalid value for {name}")
    if min_val is not None and v < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and v > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return v


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", load_error=load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if load_error:
        flash("Models not available. Please train the model first.", "danger")
        return redirect(url_for("index"))

    try:
        # Collect form inputs
        male = 1 if request.form.get("male") == "Male" else 0
        age = parse_float(request.form["age"], "Age", 20, 120)
        smoker = 1 if request.form.get("currentSmoker") == "Yes" else 0
        cigs = parse_float(request.form["cigsPerDay"], "Cigarettes per day", 0, 200)
        stroke = 1 if request.form.get("prevalentStroke") == "Yes" else 0
        hyp = 1 if request.form.get("prevalentHyp") == "Yes" else 0
        diabetes = 1 if request.form.get("diabetes") == "Yes" else 0
        chol = parse_float(request.form["totChol"], "Total Cholesterol", 50, 800)
        bmi = parse_float(request.form["BMI"], "BMI", 5, 80)
        hr = parse_float(request.form["heartRate"], "Heart Rate", 30, 220)
        glucose = parse_float(request.form["glucose"], "Glucose", 10, 500)
        pulse = parse_float(request.form["pulse_pressure"], "Pulse Pressure", 0, 300)

        X = np.array([[male, age, smoker, cigs, stroke, hyp, diabetes, chol, bmi, hr, glucose, pulse]])
        X_scaled = scaler.transform(X)

        chd_prob = float(model.predict_proba(X_scaled)[:, 1][0])
        _, final_score = predict_health_score_for_new(X_scaled, np.array([chd_prob]))
        final_score = float(final_score[0])

        chd_percent = round(chd_prob * 100, 2)
        health_score = round(np.clip(final_score, 0, 100), 2)

        plan = recommend_insurance(chd_prob, health_score)

        return render_template("result.html", details={
            "chd_prob": chd_percent,
            "health_score": health_score,
            "plan": plan,
        })

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    # Load saved metrics
    metrics = joblib.load("model_metrics.pkl")
    cluster_dist = metrics.get("cluster_distribution", {})

    # Create readable labels
    cluster_names = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk"
    }
    labels = [cluster_names.get(int(k), f"Cluster {k}") for k in cluster_dist.keys()]
    values = list(cluster_dist.values())

    # Compute percentage for each cluster
    total = sum(values)
    percentages = [round((v / total) * 100, 2) for v in values]

    # Load recent results
    df = pd.read_csv("data/processed/health_scores.csv") if os.path.exists("data/processed/health_scores.csv") else pd.DataFrame()
    health_scores = df["Final_Health_Score"].tolist()[:100] if not df.empty else []
    chd_probs = df["CHD_Probability"].tolist()[:100] if not df.empty else []

    return render_template(
        "dashboard.html",
        metrics=metrics,
        cluster_labels=labels,
        cluster_values=percentages,
        health_scores=health_scores,
        chd_probs=chd_probs
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)