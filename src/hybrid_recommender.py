from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RATINGS_DIR = PROJECT_ROOT / "data" / "ratings" / "ml-100k"

TRAIN_PATH = PROCESSED_DIR / "ratings_train.csv"
MOVIE_LENS_PATH = RATINGS_DIR / "u.item"

TFIDF_VECTOR_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
TMDB_METADATA_PATH = MODEL_DIR / "movie_metadata.joblib"


# ============================================================
# 2. LOAD THE EXISTING CONTENT MODEL
# ============================================================

print("=" * 70)
print("CINEMAMATCH AI - HYBRID RECOMMENDER")
print("=" * 70)

vectorizer = joblib.load(TFIDF_VECTOR_PATH)
tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
tmdb_metadata = joblib.load(TMDB_METADATA_PATH)

print(f"\nTMDB catalog size: {len(tmdb_metadata):,} movies")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# ============================================================
# 3. CONTENT-BASED RECOMMENDATIONS
# ============================================================

def recommend_similar_movies(movie_title, top_n=10):
    """
    Recommend TMDB movies similar to a supplied movie title.

    Duplicate titles are resolved by selecting the first matching
    catalog entry. A later version can accept a TMDB ID or year.
    """

    matches = tmdb_metadata[
        tmdb_metadata["title"].str.casefold()
        == movie_title.casefold()
    ]

    if matches.empty:
        raise ValueError(
            f"Movie '{movie_title}' was not found in the TMDB catalog."
        )

    movie_index = matches.index[0]

    similarities = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix,
    ).flatten()

    # Exclude the movie itself.
    similarities[movie_index] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = tmdb_metadata.iloc[top_indices].copy()
    results["content_similarity"] = similarities[top_indices]

    return results[
        [
            "tmdb_id",
            "title",
            "release_year",
            "genres",
            "content_similarity",
        ]
    ].reset_index(drop=True)


# ============================================================
# 4. LOAD MOVIELENS RATINGS
# ============================================================

train = pd.read_csv(TRAIN_PATH)

movie_lens_movies = pd.read_csv(
    MOVIE_LENS_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

rating_matrix = train.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
)

user_means = rating_matrix.mean(axis=1)

centered_matrix = rating_matrix.sub(
    user_means,
    axis=0,
)

similarity_values = cosine_similarity(
    centered_matrix.fillna(0)
)

user_similarity = pd.DataFrame(
    similarity_values,
    index=rating_matrix.index,
    columns=rating_matrix.index,
)

global_mean = train["rating"].mean()


# ============================================================
# 5. COLLABORATIVE RATING PREDICTION
# ============================================================

def predict_rating(user_id, movie_id, k=30):

    if user_id not in rating_matrix.index:
        return global_mean

    if movie_id not in rating_matrix.columns:
        return global_mean

    movie_ratings = rating_matrix[movie_id].dropna()

    movie_ratings = movie_ratings.drop(
        index=user_id,
        errors="ignore",
    )

    if movie_ratings.empty:
        return user_means[user_id]

    similarities = user_similarity.loc[
        user_id,
        movie_ratings.index,
    ]

    neighbors = similarities[similarities > 0].nlargest(k)

    if neighbors.empty:
        return user_means[user_id]

    neighbor_means = user_means.loc[neighbors.index]

    centered_ratings = (
        movie_ratings.loc[neighbors.index]
        - neighbor_means
    )

    denominator = neighbors.abs().sum()

    if denominator == 0:
        return user_means[user_id]

    adjustment = (
        neighbors * centered_ratings
    ).sum() / denominator

    prediction = user_means[user_id] + adjustment

    return float(np.clip(prediction, 1, 5))


# ============================================================
# 6. HYBRID RANKING FOR MOVIELENS
# ============================================================

movie_stats = (
    train.groupby("movie_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
    .reset_index()
)

minimum_rating_count = 20

movie_stats["confidence_score"] = (
    (
        movie_stats["rating_count"]
        / (movie_stats["rating_count"] + minimum_rating_count)
    )
    * movie_stats["average_rating"]
    +
    (
        minimum_rating_count
        / (movie_stats["rating_count"] + minimum_rating_count)
    )
    * global_mean
)


def recommend_for_user(user_id, top_n=10, min_ratings=20):
    """
    Generate MovieLens recommendations using collaborative
    predictions and rating confidence.

    The content model operates on TMDB IDs and is deliberately
    kept separate until a trustworthy catalog mapping is available.
    """

    if user_id not in rating_matrix.index:
        raise ValueError(
            f"User {user_id} is not present in the training data."
        )

    watched = set(
        train.loc[
            train["user_id"] == user_id,
            "movie_id",
        ]
    )

    eligible = movie_stats.loc[
        movie_stats["rating_count"] >= min_ratings,
        "movie_id",
    ]

    candidates = [
        movie_id
        for movie_id in eligible
        if movie_id not in watched
        and movie_id in rating_matrix.columns
    ]

    rows = []

    for movie_id in candidates:
        rows.append(
            {
                "movie_id": movie_id,
                "predicted_rating": predict_rating(
                    user_id,
                    movie_id,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)

    results = results.merge(
        movie_stats,
        on="movie_id",
        how="left",
    )

    results = results.merge(
        movie_lens_movies,
        on="movie_id",
        how="left",
    )

    personal_score = (
        results["predicted_rating"] - 1
    ) / 4

    confidence_score = (
        results["confidence_score"] - 1
    ) / 4

    # Collaborative + confidence ranking.
    # Content similarity is not mixed in because the catalogs
    # do not yet share a validated movie-ID mapping.
    results["hybrid_score"] = (
        0.80 * personal_score
        + 0.20 * confidence_score
    )

    results = results.sort_values(
        ["hybrid_score", "rating_count"],
        ascending=[False, False],
    )

    return results.head(top_n).reset_index(drop=True)


# ============================================================
# 7. RUN A DEMONSTRATION
# ============================================================

print("\nCONTENT-BASED DEMONSTRATION")
print("-" * 70)

try:
    content_results = recommend_similar_movies(
        "Interstellar",
        top_n=5,
    )

    print(content_results.to_string(index=False))

except ValueError as error:
    print(error)


print("\nPERSONALIZED COLLABORATIVE DEMONSTRATION")
print("-" * 70)

user_id = 196

personal_results = recommend_for_user(
    user_id=user_id,
    top_n=10,
)

print(f"\nMovieLens user: {user_id}")

print(
    personal_results[
        [
            "title",
            "predicted_rating",
            "rating_count",
            "confidence_score",
            "hybrid_score",
        ]
    ].to_string(
        index=False,
        formatters={
            "predicted_rating": "{:.2f}".format,
            "confidence_score": "{:.2f}".format,
            "hybrid_score": "{:.4f}".format,
        },
    )
)

print("\nRecommendation demonstrations completed.")