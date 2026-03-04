from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.model_train import train_chd_model
from src.health_score_kmeans import fit_kmeans_healthscore, predict_health_score_for_new
from src.recommend_insurance import recommend_insurance
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np
import joblib

# 1️⃣ Load and preprocess data
df = load_dataset()
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)

# 2️⃣ Train CHD model
model = train_chd_model(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

# ✅ Predict CHD probabilities for test set
y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

# 3️⃣ Fit KMeans-based health scoring model on test set
kmeans_results = fit_kmeans_healthscore(X_test_scaled, y_prob_test)
distance_score = kmeans_results['distance_score']
final_score = kmeans_results['final_score']
clusters = kmeans_results['labels']   # from updated return dict

# 4️⃣ Save results to CSV
results_df = pd.DataFrame({
    "CHD_Probability": y_prob_test,
    "Distance_Health_Score": distance_score,
    "Final_Health_Score": final_score,
    "Cluster": clusters
})
results_df.to_csv("data/processed/health_scores.csv", index=False)
print(" Health Scores saved to data/processed/health_scores.csv")

# 5️⃣ Recommend Insurance Plan for first test sample
sample_plan = recommend_insurance(y_prob_test[0], final_score[0])
print("\n Recommended Insurance Plan for first sample:", sample_plan)

# 6️⃣ Show example result
print("\nSample Test CHD Probability:", y_prob_test[0])
print("Distance-based Health Score:", distance_score[0])
print("Final Blended Health Score:", final_score[0])

# 7️⃣ Model Evaluation Metrics
print("\nEvaluating Model Performance...")

y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_prob_test)

accuracy = report["accuracy"] * 100
precision = report["1"]["precision"] * 100
recall = report["1"]["recall"] * 100
f1_score = report["1"]["f1-score"] * 100

print(f"\nModel Performance Metrics:")
print(f"   Accuracy:  {accuracy:.2f}%")
print(f"   Precision: {precision:.2f}%")
print(f"   Recall:    {recall:.2f}%")
print(f"   F1-Score:  {f1_score:.2f}%")
print(f"   ROC-AUC:   {roc_auc:.3f}")

# 8️⃣ Cluster Distribution
cluster_counts = results_df["Cluster"].value_counts().sort_index()
total_samples = len(results_df)
print("\nTrue Cluster Distribution (from KMeans):")
for cluster, count in cluster_counts.items():
    pct = (count / total_samples) * 100
    print(f"   Cluster {cluster}: {count} samples ({pct:.2f}%)")

# Save performance + cluster info for dashboard use
metrics = {
    "accuracy": round(accuracy, 2),
    "precision": round(precision, 2),
    "recall": round(recall, 2),
    "f1_score": round(f1_score, 2),
    "roc_auc": round(roc_auc, 3),
    "cluster_distribution": cluster_counts.to_dict()
}
joblib.dump(metrics, "model_metrics.pkl")

print("\n Model metrics saved to model_metrics.pkl")
print("\n Pipeline executed successfully!")