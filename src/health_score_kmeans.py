# health_score_kmeans.py
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

def fit_kmeans_healthscore(X_scaled, chd_prob, n_clusters=3, random_state=42):
    """
    Fit KMeans on X_scaled and derive a distance-based health score (0-100).
    X_scaled: ndarray (n_samples, n_features) - scaled with same scaler used in pipeline
    chd_prob: ndarray (n_samples,) - CHD probability from classifier (0..1)
    Returns:
      dict with models and arrays:
        {
          'kmeans': fitted KMeans,
          'healthy_cluster': int,
          'centroids': ndarray,
          'distance_score': ndarray (0..100),
          'final_score': ndarray (0..100)  # after optional blending with chd_prob
          'clusters': labels,
	  'meta': meta
        }
    """
    # 1) Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # 2) Identify healthiest cluster by lowest mean CHD probability
    cluster_chd_mean = []
    for k in range(n_clusters):
        mask = (labels == k)
        if mask.sum() == 0:
            cluster_chd_mean.append(np.inf)
        else:
            cluster_chd_mean.append(np.mean(chd_prob[mask]))
    healthy_cluster = int(np.argmin(cluster_chd_mean))

    # 3) Compute distance to healthy centroid for every sample
    healthy_centroid = centroids[healthy_cluster].reshape(1, -1)
    dists = np.linalg.norm(X_scaled - healthy_centroid, axis=1)  # Euclidean

    # 4) Robust clip to reduce outlier effect (2nd-98th percentile)
    lo, hi = np.percentile(dists, 2), np.percentile(dists, 98)
    dists_clipped = np.clip(dists, lo, hi)

    # 5) Normalize and invert to 0..100 (closer -> higher score)
    scaler = MinMaxScaler(feature_range=(0, 100))
    dist_norm = scaler.fit_transform(dists_clipped.reshape(-1, 1)).ravel()
    distance_score = 100 - dist_norm  # invert: small distance -> near 100

    # 6) Optional: blend with CHD probability for final score
    alpha = 0.6  # weight for distance_score; (1-alpha) for CHD-prob-based score
    chd_component = 100 * (1 - chd_prob)  # 0..100 (higher better)
    final_score = alpha * distance_score + (1 - alpha) * chd_component

    # 7) Save models
    joblib.dump(kmeans, "kmeans_health.pkl")
    joblib.dump(scaler, "kmeans_distance_scaler.pkl")
    # Save meta info
    meta = {
        'n_clusters': n_clusters,
        'healthy_cluster': healthy_cluster,
        'cluster_chd_mean': cluster_chd_mean,
        'alpha': alpha
    }
    joblib.dump(meta, "kmeans_meta.pkl")

    print("Fitted KMeans health scoring (n_clusters=%d). Healthy cluster = %d" % (n_clusters, healthy_cluster))
    return {
        'kmeans': kmeans,
        'healthy_cluster': healthy_cluster,
        'centroids': centroids,
        'distance_score': distance_score,
        'final_score': final_score,
        'labels': labels,
        'meta': meta
    }


def predict_health_score_for_new(X_new_scaled, chd_prob_new, kmeans_path="kmeans_health.pkl",
                                 scaler_path="kmeans_distance_scaler.pkl", meta_path="kmeans_meta.pkl"):
    """
    For a new sample(s), compute distance-based score and blended final score.
    X_new_scaled: ndarray (m_samples, n_features)
    chd_prob_new: ndarray (m_samples,)  # predicted CHD probability
    """
    kmeans = joblib.load(kmeans_path)
    dist_scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    healthy_cluster = meta['healthy_cluster']
    alpha = meta.get('alpha', 0.6)

    healthy_centroid = kmeans.cluster_centers_[healthy_cluster].reshape(1, -1)
    dists_new = np.linalg.norm(X_new_scaled - healthy_centroid, axis=1)

    # Clip using same percentile ranges? We don't have original bounds here;
    # instead scale relative to fitted scaler's data range via transform after reshape.
    # To avoid errors, we will transform after clipping to reasonable range:
    # Here we assume dist_scaler was fitted on clipped distances of training data.
    dists_new_clipped = np.clip(dists_new, 0, None)  # ensure non-negative
    dist_norm_new = dist_scaler.transform(dists_new_clipped.reshape(-1,1)).ravel()
    distance_score_new = 100 - dist_norm_new

    chd_component_new = 100 * (1 - chd_prob_new)
    final_score_new = alpha * distance_score_new + (1 - alpha) * chd_component_new

    return distance_score_new, final_score_new