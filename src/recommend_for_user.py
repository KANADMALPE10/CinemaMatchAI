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

print("=" * 65)
print("CINEMAMATCH AI - PERSONALIZED MOVIE RECOMMENDATIONS")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1, 2],
    names=["movie_id", "title", "release_date"],
)

# --------------------------------------------------
# 3. BUILD THE RATING MATRIX
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

centered_filled = centered_matrix.fillna(0)

similarity_matrix = cosine_similarity(
    centered_filled
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index,
)

global_mean = train["rating"].mean()

# --------------------------------------------------
# 4. PREDICT A USER'S RATING FOR A MOVIE
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

    positive_similarities = similarities[
        similarities > 0
    ]

    if positive_similarities.empty:
        return user_means[user_id]

    neighbors = positive_similarities.nlargest(k)

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


# --------------------------------------------------
# 5. GENERATE RECOMMENDATIONS
# --------------------------------------------------

def recommend_movies(user_id, top_n=10):

    if user_id not in rating_matrix.index:
        raise ValueError(
            f"User {user_id} is not present in the training data."
        )

    # Movies already rated by this user in the training data
    watched_movies = set(
        train.loc[
            train["user_id"] == user_id,
            "movie_id",
        ]
    )

    # Candidate movies are those represented in training.
    candidate_movies = [
        movie_id
        for movie_id in rating_matrix.columns
        if movie_id not in watched_movies
    ]

    # Predict ratings for candidate movies
    recommendations = []

    for movie_id in candidate_movies:

        predicted_rating = predict_rating(
            user_id,
            movie_id,
        )

        recommendations.append(
            {
                "movie_id": movie_id,
                "predicted_rating": predicted_rating,
            }
        )

    result = pd.DataFrame(recommendations)

    result = result.merge(
        movies,
        on="movie_id",
        how="left",
    )

    result = result.sort_values(
        "predicted_rating",
        ascending=False,
    )

    return result.head(top_n).reset_index(drop=True)


# --------------------------------------------------
# 6. DISPLAY RECOMMENDATIONS
# --------------------------------------------------

# Example user ID from MovieLens
USER_ID = 196

recommendations = recommend_movies(
    user_id=USER_ID,
    top_n=10,
)

print(f"\nRecommendations for MovieLens user {USER_ID}")
print("-" * 65)

print(
    recommendations[
        ["title", "predicted_rating"]
    ].to_string(
        index=False,
        formatters={
            "predicted_rating": "{:.2f}".format
        },
    )
)

print("\nRecommendation generation completed.")