from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "ratings_train_consolidated.csv")
test = pd.read_csv(DATA_DIR / "ratings_test_consolidated.csv")

# Create user-item matrix using training ratings only.
user_item = train.pivot(
    index="user_id",
    columns="tmdb_id",
    values="rating",
)

global_mean = train["rating"].mean()
user_means = user_item.mean(axis=1)
movie_means = user_item.mean(axis=0)

# Center ratings by user mean; missing ratings become zero.
centered = user_item.sub(user_means, axis=0).fillna(0)

# Calculate user-to-user cosine similarities.
similarities = cosine_similarity(centered)

user_ids = user_item.index.to_numpy()
movie_ids = user_item.columns.to_numpy()

user_index = {uid: i for i, uid in enumerate(user_ids)}
movie_index = {mid: i for i, mid in enumerate(movie_ids)}

# Use the same baseline fallback for unseen users or movies.
def baseline_prediction(user_id, movie_id):
    movie_mean = movie_means.get(movie_id, np.nan)

    if pd.notna(movie_mean):
        return float(movie_mean)

    user_mean = user_means.get(user_id, np.nan)

    if pd.notna(user_mean):
        return float(user_mean)

    return float(global_mean)


def predict_rating(user_id, movie_id, k=30):
    if user_id not in user_index or movie_id not in movie_index:
        return baseline_prediction(user_id, movie_id)

    u_idx = user_index[user_id]
    m_idx = movie_index[movie_id]

    # Only users who rated this movie can contribute.
    rated_mask = user_item.iloc[:, m_idx].notna().to_numpy(copy=True)

    # Exclude the target user.
    rated_mask[u_idx] = False

    candidate_indices = np.flatnonzero(rated_mask)

    if len(candidate_indices) == 0:
        return baseline_prediction(user_id, movie_id)

    candidate_sims = similarities[u_idx, candidate_indices]

    # Use positive-similarity neighbors only.
    positive_mask = candidate_sims > 0
    candidate_indices = candidate_indices[positive_mask]
    candidate_sims = candidate_sims[positive_mask]

    if len(candidate_indices) == 0:
        return baseline_prediction(user_id, movie_id)

    # Select the strongest neighbors.
    top_positions = np.argsort(candidate_sims)[-k:]
    neighbors = candidate_indices[top_positions]
    weights = candidate_sims[top_positions]

    neighbor_ratings = user_item.iloc[neighbors, m_idx].to_numpy()
    neighbor_means = user_means.iloc[neighbors].to_numpy()

    # Mean-centered weighted prediction.
    deviations = neighbor_ratings - neighbor_means
    denominator = np.abs(weights).sum()

    if denominator == 0:
        return baseline_prediction(user_id, movie_id)

    prediction = user_means.iloc[u_idx] + np.dot(weights, deviations) / denominator

    return float(np.clip(prediction, 1, 5))


print("Generating collaborative-filtering predictions...")

predictions = np.array([
    predict_rating(row.user_id, row.tmdb_id)
    for row in test.itertuples(index=False)
])

actual = test["rating"].to_numpy()

cf_mae = mean_absolute_error(actual, predictions)
cf_rmse = np.sqrt(mean_squared_error(actual, predictions))

# Evaluate the movie-average baseline on the same test rows.
baseline_predictions = test["tmdb_id"].map(movie_means)
baseline_predictions = baseline_predictions.fillna(global_mean).clip(1, 5)

baseline_mae = mean_absolute_error(actual, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_predictions))

print("\n" + "=" * 55)
print("CINEMAMATCH AI - COLLABORATIVE FILTERING EVALUATION")
print("=" * 55)

print(f"Test ratings: {len(test):,}")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Collaborative-filtering MAE: {cf_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Collaborative-filtering RMSE: {cf_rmse:.4f}")

print("\nMAE improvement:", f"{baseline_mae - cf_mae:.4f}")
print("RMSE improvement:", f"{baseline_rmse - cf_rmse:.4f}")
