def recommend_insurance(chd_prob, health_score):
    if chd_prob > 0.2 or health_score < 50:
        return "Premium Plan"
    elif health_score < 70:
        return "Standard Plan"
    else:
        return "Basic Plan"