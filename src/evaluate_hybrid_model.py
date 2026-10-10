from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PATHS AND DATA
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "ratings_train_hybrid.csv"
TEST_PATH = DATA_DIR / "ratings_test_hybrid.csv"

TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
METADATA_PATH = MODEL_DIR / "movie_metadata.joblib"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
metadata = joblib.load(METADATA_PATH).reset_index(drop=True)

metadata["tmdb_id"] = metadata["tmdb_id"].astype(int)

tmdb_to_index = {
    int(tmdb_id): index
    for index, tmdb_id in enumerate(metadata["tmdb_id"])
}

print("=" * 65)
print("CINEMAMATCH AI - HYBRID MODEL EVALUATION")
print("=" * 65)


# ============================================================
# 2. COLLABORATIVE-FILTERING MODEL
# ============================================================

rating_matrix = train.pivot(
    index="user_id",
    columns="tmdb_id",
    values="rating",
)

user_means = rating_matrix.mean(axis=1)
movie_means = rating_matrix.mean(axis=0)
global_mean = float(train["rating"].mean())

centered = rating_matrix.sub(user_means, axis=0).fillna(0)

similarities = cosine_similarity(centered)

user_ids = rating_matrix.index.to_numpy()

user_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}


def baseline_prediction(user_id, movie_id):
    movie_mean = movie_means.get(movie_id, np.nan)

    if pd.notna(movie_mean):
        return float(movie_mean)

    user_mean = user_means.get(user_id, np.nan)

    if pd.notna(user_mean):
        return float(user_mean)

    return global_mean


def predict_rating(user_id, movie_id, k=30):
    if user_id not in user_index or movie_id not in rating_matrix.columns:
        return baseline_prediction(user_id, movie_id)

    u_idx = user_index[user_id]
    movie_ratings = rating_matrix[movie_id]

    rated_mask = movie_ratings.notna().to_numpy(copy=True)
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

    ratings = movie_ratings.iloc[neighbors].to_numpy()
    means = user_means.iloc[neighbors].to_numpy()

    denominator = np.abs(weights).sum()

    if denominator == 0:
        return baseline_prediction(user_id, movie_id)

    prediction = (
        user_means.iloc[u_idx]
        + np.dot(weights, ratings - means) / denominator
    )

    return float(np.clip(prediction, 1, 5))


# ============================================================
# 3. CONTENT SIMILARITY
# ============================================================

def similarity_to_liked_movies(user_id, candidate_id):
    if candidate_id not in tmdb_to_index:
        return 0.0

    liked_movies = train.loc[
        (train["user_id"] == user_id)
        & (train["rating"] >= 4),
        "tmdb_id",
    ].tolist()

    liked_indices = [
        tmdb_to_index[movie_id]
        for movie_id in liked_movies
        if movie_id in tmdb_to_index
    ]

    if not liked_indices:
        return 0.0

    candidate_index = tmdb_to_index[candidate_id]

    scores = cosine_similarity(
        tfidf_matrix[candidate_index],
        tfidf_matrix[liked_indices],
    ).ravel()

    return float(scores.max())


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions for the test set...")

rows = []

for row in test.itertuples(index=False):
    user_id = int(row.user_id)
    movie_id = int(row.tmdb_id)

    collaborative = predict_rating(user_id, movie_id)

    content = similarity_to_liked_movies(user_id, movie_id)

    # Normalize predicted ratings to [0, 1].
    collaborative_normalized = (collaborative - 1) / 4

    # Content similarity is already in [0, 1].
    # This is a deliberately simple initial hybrid.
    hybrid = (
        0.80 * collaborative_normalized
        + 0.20 * content
    )

    # Convert the hybrid score to the rating scale.
    hybrid_prediction = 1 + 4 * hybrid

    rows.append({
        "user_id": user_id,
        "tmdb_id": movie_id,
        "actual_rating": float(row.rating),
        "baseline_prediction": baseline_prediction(user_id, movie_id),
        "collaborative_prediction": collaborative,
        "content_similarity": content,
        "hybrid_prediction": hybrid_prediction,
    })

results = pd.DataFrame(rows)


# ============================================================
# 5. EVALUATION
# ============================================================

actual = results["actual_rating"]

models = {
    "Baseline": results["baseline_prediction"],
    "Collaborative Filtering": results["collaborative_prediction"],
    "Hybrid": results["hybrid_prediction"],
}

print("\n" + "=" * 65)
print("EVALUATION RESULTS")
print("=" * 65)

for name, predictions in models.items():
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

print(f"\nTest ratings evaluated: {len(results):,}")

output_path = DATA_DIR / "hybrid_evaluation_predictions.csv"
results.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")