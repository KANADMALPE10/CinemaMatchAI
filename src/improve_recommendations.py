import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ratings_train.csv"
)

MOVIES_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.item"
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

# --------------------------------------------------
# 3. BUILD THE USER-MOVIE MATRIX
# --------------------------------------------------

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

similarity_matrix = cosine_similarity(
    centered_matrix.fillna(0)
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index,
)

global_mean = train["rating"].mean()

# --------------------------------------------------
# 4. COLLABORATIVE-FILTERING PREDICTION
# --------------------------------------------------

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

    similarities = similarity_df.loc[
        user_id,
        movie_ratings.index,
    ]

    neighbors = similarities[similarities > 0].nlargest(k)

    if neighbors.empty:
        return user_means[user_id]

    neighbor_means = user_means.loc[neighbors.index]

    centered_ratings = (
        movie_ratings.loc[neighbors.index] - neighbor_means
    )

    denominator = neighbors.abs().sum()

    if denominator == 0:
        return user_means[user_id]

    adjustment = (
        neighbors * centered_ratings
    ).sum() / denominator

    return float(
        np.clip(user_means[user_id] + adjustment, 1, 5)
    )


# --------------------------------------------------
# 5. CONFIDENCE-ADJUSTED MOVIE STATISTICS
# --------------------------------------------------

movie_stats = (
    train.groupby("movie_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
    .reset_index()
)

movie_stats["bayesian_score"] = (
    movie_stats["rating_count"]
    / (movie_stats["rating_count"] + 20)
    * movie_stats["average_rating"]
    +
    20
    / (movie_stats["rating_count"] + 20)
    * global_mean
)

# --------------------------------------------------
# 6. GENERATE HYBRID RANKING
# --------------------------------------------------

def recommend_movies(user_id, top_n=10, min_ratings=20):

    if user_id not in rating_matrix.index:

        raise ValueError(

            f"User {user_id} is not present in the training data."

        )

    # Exclude movies the user has already rated.

    watched = set(

        train.loc[

            train["user_id"] == user_id,

            "movie_id",

        ]

    )

    # Only recommend movies with sufficient rating history.

    eligible_movies = set(

        movie_stats.loc[

            movie_stats["rating_count"] >= min_ratings,

            "movie_id",

        ]

    )

    candidates = [

        movie_id

        for movie_id in rating_matrix.columns

        if movie_id not in watched

        and movie_id in eligible_movies

    ]

    rows = []

    for movie_id in candidates:

        predicted_rating = predict_rating(

            user_id,

            movie_id,

        )

        rows.append(

            {

                "movie_id": movie_id,

                "predicted_rating": predicted_rating,

            }

        )

    recommendations = pd.DataFrame(rows)

    if recommendations.empty:

        return recommendations

    recommendations = recommendations.merge(

        movie_stats,

        on="movie_id",

        how="left",

    )

    recommendations = recommendations.merge(

        movies,

        on="movie_id",

        how="left",

    )

    # Normalize personalized predictions to a 0–1 scale.

    personal_score = (

        recommendations["predicted_rating"] - 1

    ) / 4

    # Normalize confidence-adjusted ratings to a 0–1 scale.

    reliability_score = (

        recommendations["bayesian_score"] - 1

    ) / 4

    # Combine personal preference and rating reliability.

    recommendations["hybrid_score"] = (

        0.75 * personal_score

        + 0.25 * reliability_score

    )

    recommendations = recommendations.sort_values(

        ["hybrid_score", "rating_count"],

        ascending=[False, False],

    )

    return recommendations.head(top_n).reset_index(drop=True)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

USER_ID = 196

recommendations = recommend_movies(
    user_id=USER_ID,
    top_n=10,
)

print("=" * 75)
print("CINEMAMATCH AI - IMPROVED PERSONALIZED RECOMMENDATIONS")
print("=" * 75)

print(f"\nUser ID: {USER_ID}")
print("\nTOP 10 RECOMMENDATIONS")
print("-" * 75)

print(
    recommendations[
        [
            "title",
            "predicted_rating",
            "rating_count",
            "bayesian_score",
            "hybrid_score",
        ]
    ].to_string(
        index=False,
        formatters={
            "predicted_rating": "{:.2f}".format,
            "bayesian_score": "{:.2f}".format,
            "hybrid_score": "{:.4f}".format,
        },
    )
)

print("\nImproved recommendation ranking completed.")