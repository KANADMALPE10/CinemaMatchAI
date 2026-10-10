from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "ratings_train_consolidated.csv"
MAPPING_PATH = DATA_DIR / "movie_catalog_mapping_approved.csv"

TFIDF_VECTOR_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
METADATA_PATH = MODEL_DIR / "movie_metadata.joblib"


# ============================================================
# 2. LOAD DATA AND CONTENT MODEL
# ============================================================

print("=" * 65)
print("CINEMAMATCH AI - HYBRID RECOMMENDER V2")
print("=" * 65)

train = pd.read_csv(TRAIN_PATH)
mapping = pd.read_csv(MAPPING_PATH)

vectorizer = joblib.load(TFIDF_VECTOR_PATH)
tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
metadata = joblib.load(METADATA_PATH)

metadata = metadata.reset_index(drop=True)

metadata["tmdb_id"] = metadata["tmdb_id"].astype(int)

print(f"\nTMDB catalog: {len(metadata):,} movies")
print(f"Consolidated ratings: {len(train):,}")
print(f"Users: {train['user_id'].nunique():,}")


# ============================================================
# 3. BUILD CONTENT-BASED SIMILARITY
# ============================================================

tmdb_to_index = {
    int(tmdb_id): index
    for index, tmdb_id in enumerate(metadata["tmdb_id"])
}


def get_content_similarity(movie_id, candidate_ids):
    """Calculate TF-IDF similarity between one movie and candidates."""

    if movie_id not in tmdb_to_index:
        return {}

    source_index = tmdb_to_index[movie_id]

    candidate_indices = [
        tmdb_to_index[movie_id]
        for movie_id in candidate_ids
        if movie_id in tmdb_to_index
    ]

    if not candidate_indices:
        return {}

    scores = cosine_similarity(
        tfidf_matrix[source_index],
        tfidf_matrix[candidate_indices],
    ).ravel()

    return {
        int(candidate_id): float(score)
        for candidate_id, score in zip(candidate_ids, scores)
        if candidate_id in tmdb_to_index
    }


# ============================================================
# 4. BUILD COLLABORATIVE-FILTERING MODEL
# ============================================================

rating_matrix = train.pivot(
    index="user_id",
    columns="tmdb_id",
    values="rating",
)

user_means = rating_matrix.mean(axis=1)
movie_means = rating_matrix.mean(axis=0)

global_mean = float(train["rating"].mean())

centered = rating_matrix.sub(
    user_means,
    axis=0,
).fillna(0)

similarities = cosine_similarity(centered)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

user_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

movie_stats = (
    train.groupby("tmdb_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
)

minimum_rating_count = 20

movie_stats["confidence_score"] = (
    movie_stats["rating_count"]
    / (movie_stats["rating_count"] + minimum_rating_count)
    * movie_stats["average_rating"]
    +
    minimum_rating_count
    / (movie_stats["rating_count"] + minimum_rating_count)
    * global_mean
)


def predict_rating(user_id, movie_id, k=30):
    """Predict a user's rating using similar users."""

    if user_id not in user_index:
        return global_mean

    if movie_id not in movie_index:
        return float(movie_stats.loc[movie_id, "confidence_score"])

    user_position = user_index[user_id]
    movie_position = movie_index[movie_id]

    rated_mask = rating_matrix.iloc[:, movie_position].notna().to_numpy(
        copy=True
    )
    rated_mask[user_position] = False

    neighbor_positions = np.flatnonzero(rated_mask)

    if len(neighbor_positions) == 0:
        return float(user_means.loc[user_id])

    neighbor_similarities = similarities[
        user_position,
        neighbor_positions,
    ]

    positive = neighbor_similarities > 0

    neighbor_positions = neighbor_positions[positive]
    neighbor_similarities = neighbor_similarities[positive]

    if len(neighbor_positions) == 0:
        return float(user_means.loc[user_id])

    top = np.argsort(neighbor_similarities)[-k:]

    neighbors = neighbor_positions[top]
    weights = neighbor_similarities[top]

    ratings = rating_matrix.iloc[neighbors, movie_position].to_numpy()
    means = user_means.iloc[neighbors].to_numpy()

    denominator = np.abs(weights).sum()

    if denominator == 0:
        return float(user_means.loc[user_id])

    prediction = (
        user_means.loc[user_id]
        + np.dot(weights, ratings - means) / denominator
    )

    return float(np.clip(prediction, 1, 5))


# ============================================================
# 5. GENERATE PERSONALIZED HYBRID RECOMMENDATIONS
# ============================================================

def recommend_for_user(
    user_id,
    top_n=10,
    min_ratings=20,
    content_weight=0.30,
):
    """
    Combine collaborative predictions and content similarity.

    The content component represents similarity to movies the user
    has rated positively. The collaborative component represents
    personalized rating predictions.
    """

    if user_id not in rating_matrix.index:
        raise ValueError(
            f"User {user_id} is not present in the training data."
        )

    watched = set(
        train.loc[train["user_id"] == user_id, "tmdb_id"]
    )

    eligible = movie_stats.index[
        movie_stats["rating_count"] >= min_ratings
    ]

    candidates = [
        int(movie_id)
        for movie_id in eligible
        if movie_id not in watched
        and movie_id in tmdb_to_index
    ]

    if not candidates:
        return pd.DataFrame()

    # Identify movies the user rated positively.
    user_ratings = train[train["user_id"] == user_id]

    liked = user_ratings[user_ratings["rating"] >= 4]["tmdb_id"].tolist()

    # Compute each candidate's similarity to the user's liked movies.
    content_scores = {}

    for candidate in candidates:
        scores = [
            get_content_similarity(liked_movie, [candidate]).get(candidate, 0.0)
            for liked_movie in liked
        ]

        content_scores[candidate] = max(scores, default=0.0)

    rows = []

    for movie_id in candidates:
        predicted = predict_rating(user_id, movie_id)

        confidence = float(
            (movie_stats.loc[movie_id, "confidence_score"] - 1) / 4
        )

        collaborative = (predicted - 1) / 4
        content = content_scores[movie_id]

        hybrid_score = (
            (1 - content_weight) * collaborative
            + content_weight * content
        )

        rows.append({
            "tmdb_id": movie_id,
            "predicted_rating": predicted,
            "content_similarity": content,
            "confidence_score": confidence,
            "hybrid_score": hybrid_score,
        })

    results = pd.DataFrame(rows)

    results = results.merge(
        metadata[
            [
                "tmdb_id",
                "title",
                "release_year",
                "genres",
                "overview",
            ]
        ],
        on="tmdb_id",
        how="left",
    )

    results = results.sort_values(
        ["hybrid_score", "confidence_score"],
        ascending=[False, False],
    )

    return results.head(top_n).reset_index(drop=True)


# ============================================================
# 6. DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    demo_user = 196

    recommendations = recommend_for_user(
        user_id=demo_user,
        top_n=10,
    )

    print(f"\nRecommendations for user {demo_user}:\n")

    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(
            recommendations[
                [
                    "title",
                    "release_year",
                    "predicted_rating",
                    "content_similarity",
                    "hybrid_score",
                ]
            ].to_string(
                index=False,
                formatters={
                    "predicted_rating": "{:.2f}".format,
                    "content_similarity": "{:.3f}".format,
                    "hybrid_score": "{:.4f}".format,
                },
            )
        )