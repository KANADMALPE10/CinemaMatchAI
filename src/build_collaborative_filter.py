import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "ratings_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "ratings_test.csv"

print("=" * 65)
print("CINEMAMATCH AI - USER-BASED COLLABORATIVE FILTERING")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"\nTraining ratings: {len(train):,}")
print(f"Testing ratings: {len(test):,}")

# --------------------------------------------------
# 3. CREATE THE USER-MOVIE RATING MATRIX
# --------------------------------------------------

rating_matrix = train.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
)

print(f"\nRating matrix shape: {rating_matrix.shape}")

# --------------------------------------------------
# 4. CENTER RATINGS BY USER
# --------------------------------------------------

# Users who give mostly high ratings and users who give
# mostly low ratings should not automatically be treated
# as having different tastes.

user_means = rating_matrix.mean(axis=1)

centered_matrix = rating_matrix.sub(user_means, axis=0)

# Missing ratings represent unknown preferences.
# Replace missing centered values with zero for similarity.
centered_filled = centered_matrix.fillna(0)

# --------------------------------------------------
# 5. CALCULATE USER SIMILARITIES
# --------------------------------------------------

print("\nCalculating user similarities...")

similarity_matrix = cosine_similarity(centered_filled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index,
)

print("User similarity matrix created.")

# --------------------------------------------------
# 6. PREPARE A PREDICTION FUNCTION
# --------------------------------------------------

global_mean = train["rating"].mean()

def predict_rating(user_id, movie_id, k=30):
    """
    Predict a user's rating for a movie using the ratings
    of the most similar users who rated that movie.
    """

    # The user or movie may be absent from training.
    if user_id not in rating_matrix.index:
        return global_mean

    if movie_id not in rating_matrix.columns:
        return global_mean

    # Ratings for this movie from all users.
    movie_ratings = rating_matrix[movie_id].dropna()

    # Exclude the target user if they rated this movie.
    movie_ratings = movie_ratings.drop(
        index=user_id,
        errors="ignore",
    )

    if movie_ratings.empty:
        return user_means[user_id]

    # Similarities between the target user and users
    # who rated the target movie.
    similarities = similarity_df.loc[
        user_id, movie_ratings.index
    ]

    # Keep only positively similar users.
    positive_similarities = similarities[
        similarities > 0
    ]

    if positive_similarities.empty:
        return user_means[user_id]

    # Select the k most similar users.
    neighbors = positive_similarities.nlargest(k)

    # Ratings centered around each neighbor's own mean.
    neighbor_means = user_means.loc[neighbors.index]

    neighbor_centered_ratings = (
        movie_ratings.loc[neighbors.index] - neighbor_means
    )

    # Similarity-weighted deviation from the user's mean.
    denominator = neighbors.abs().sum()

    if denominator == 0:
        return user_means[user_id]

    adjustment = (
        neighbors * neighbor_centered_ratings
    ).sum() / denominator

    prediction = user_means[user_id] + adjustment

    return float(np.clip(prediction, 1, 5))


# --------------------------------------------------
# 7. EVALUATE ON THE TEST SET
# --------------------------------------------------

# Evaluate a subset first to keep this initial experiment
# manageable while checking that the algorithm works.

EVALUATION_SIZE = min(3000, len(test))

evaluation = test.sample(
    n=EVALUATION_SIZE,
    random_state=42,
).copy()

print(
    f"\nEvaluating {len(evaluation):,} test ratings..."
)

predictions = []

for row in evaluation.itertuples(index=False):
    prediction = predict_rating(
        row.user_id,
        row.movie_id,
    )
    predictions.append(prediction)

evaluation["predicted_rating"] = predictions

actual = evaluation["rating"]
predicted = evaluation["predicted_rating"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\nCOLLABORATIVE FILTERING RESULTS")
print("-" * 45)
print(f"Evaluation ratings: {len(evaluation):,}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# --------------------------------------------------
# 8. COMPARE AGAINST THE BASELINE
# --------------------------------------------------

baseline_movie_means = train.groupby("movie_id")["rating"].mean()

baseline_predictions = (
    evaluation["movie_id"]
    .map(baseline_movie_means)
    .fillna(global_mean)
    .clip(1, 5)
)

baseline_mae = mean_absolute_error(
    actual,
    baseline_predictions,
)

baseline_rmse = np.sqrt(
    mean_squared_error(actual, baseline_predictions)
)

print("\nBASELINE VS COLLABORATIVE FILTERING")
print("-" * 45)
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Collaborative MAE: {mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Collaborative RMSE: {rmse:.4f}")

mae_change = baseline_mae - mae
rmse_change = baseline_rmse - rmse

print(f"\nMAE improvement: {mae_change:+.4f}")
print(f"RMSE improvement: {rmse_change:+.4f}")

print("\nCollaborative filtering experiment completed.")