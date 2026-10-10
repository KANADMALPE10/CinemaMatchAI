from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "ratings_train_hybrid.csv")
test = pd.read_csv(DATA_DIR / "ratings_test_hybrid.csv")

# Build the user-item matrix using training data only.
rating_matrix = train.pivot(
    index="user_id",
    columns="tmdb_id",
    values="rating",
)

user_means = rating_matrix.mean(axis=1)
movie_means = rating_matrix.mean(axis=0)
global_mean = float(train["rating"].mean())

# Center ratings around each user's mean.
centered = rating_matrix.sub(user_means, axis=0).fillna(0)

# Compute user similarities.
similarities = cosine_similarity(centered)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

user_index = {uid: i for i, uid in enumerate(user_ids)}
movie_index = {mid: i for i, mid in enumerate(movie_ids)}


def baseline_prediction(user_id, movie_id):
    movie_mean = movie_means.get(movie_id, np.nan)

    if pd.notna(movie_mean):
        return float(movie_mean)

    user_mean = user_means.get(user_id, np.nan)

    if pd.notna(user_mean):
        return float(user_mean)

    return global_mean


def predict_rating(user_id, movie_id, k=30):
    if user_id not in user_index or movie_id not in movie_index:
        return baseline_prediction(user_id, movie_id)

    u_idx = user_index[user_id]
    m_idx = movie_index[movie_id]

    rated_mask = rating_matrix.iloc[:, m_idx].notna().to_numpy(
        copy=True
    )
    rated_mask[u_idx] = False

    neighbors = np.flatnonzero(rated_mask)

    if len(neighbors) == 0:
        return baseline_prediction(user_id, movie_id)

    neighbor_sims = similarities[u_idx, neighbors]

    positive = neighbor_sims > 0
    neighbors = neighbors[positive]
    neighbor_sims = neighbor_sims[positive]

    if len(neighbors) == 0:
        return baseline_prediction(user_id, movie_id)

    top = np.argsort(neighbor_sims)[-k:]
    neighbors = neighbors[top]
    weights = neighbor_sims[top]

    neighbor_ratings = rating_matrix.iloc[neighbors, m_idx].to_numpy()
    neighbor_means = user_means.iloc[neighbors].to_numpy()

    denominator = np.abs(weights).sum()

    if denominator == 0:
        return baseline_prediction(user_id, movie_id)

    prediction = (
        user_means.iloc[u_idx]
        + np.dot(weights, neighbor_ratings - neighbor_means)
        / denominator
    )

    return float(np.clip(prediction, 1, 5))


print("Generating collaborative-filtering predictions...")

actual = test["rating"].to_numpy()

predictions = np.array([
    predict_rating(row.user_id, row.tmdb_id)
    for row in test.itertuples(index=False)
])

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

# Calculate baseline metrics on exactly the same test rows.
baseline_predictions = np.array([
    baseline_prediction(row.user_id, row.tmdb_id)
    for row in test.itertuples(index=False)
])

baseline_mae = mean_absolute_error(actual, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(actual, baseline_predictions)
)

print("\n" + "=" * 60)
print("CINEMAMATCH AI - CORRECTED COLLABORATIVE FILTERING")
print("=" * 60)

print(f"Test ratings: {len(test):,}")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Collaborative-filtering MAE: {mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Collaborative-filtering RMSE: {rmse:.4f}")
print(f"MAE improvement: {baseline_mae - mae:.4f}")
print(f"RMSE improvement: {baseline_rmse - rmse:.4f}")