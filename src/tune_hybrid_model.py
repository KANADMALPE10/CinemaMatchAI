from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "ratings_train_hybrid.csv"
TEST_PATH = DATA_DIR / "ratings_test_hybrid.csv"

TFIDF_PATH = MODEL_DIR / "tfidf_matrix.joblib"
METADATA_PATH = MODEL_DIR / "movie_metadata.joblib"

RANDOM_STATE = 42
TOP_K_NEIGHBORS = 30


# ============================================================
# 2. LOAD DATA AND EXISTING CONTENT-MODEL ARTIFACTS
# ============================================================

print("=" * 65)
print("CINEMAMATCH AI - HYBRID MODEL TUNING")
print("=" * 65)

ratings = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

tfidf_matrix = joblib.load(TFIDF_PATH)
metadata = joblib.load(METADATA_PATH)

# Normalize identifier types.
for df in (ratings, test):
    df["user_id"] = df["user_id"].astype(int)
    df["tmdb_id"] = df["tmdb_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

metadata["tmdb_id"] = metadata["tmdb_id"].astype(int)

movie_to_index = {
    int(tmdb_id): index
    for index, tmdb_id in enumerate(metadata["tmdb_id"])
}

print(f"\nTraining interactions: {len(ratings):,}")
print(f"Test interactions: {len(test):,}")
print(f"Movies in content model: {len(movie_to_index):,}")


# ============================================================
# 3. BUILD A COLLABORATIVE-FILTERING MODEL
# ============================================================

def build_cf_model(train_df):
    """
    Build a user-based collaborative-filtering model.

    Ratings are centered around each user's average rating.
    Similarities are calculated using cosine similarity.
    """

    global_mean = float(train_df["rating"].mean())

    user_means = train_df.groupby("user_id")["rating"].mean()
    movie_means = train_df.groupby("tmdb_id")["rating"].mean()

    rating_matrix = train_df.pivot_table(
        index="user_id",
        columns="tmdb_id",
        values="rating",
        aggfunc="mean"
    )

    user_means_for_matrix = rating_matrix.mean(axis=1)

    centered_matrix = rating_matrix.sub(
        user_means_for_matrix,
        axis=0
    ).fillna(0.0)

    similarity_matrix = cosine_similarity(centered_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

    return {
        "global_mean": global_mean,
        "user_means": user_means,
        "movie_means": movie_means,
        "rating_matrix": rating_matrix,
        "user_means_for_matrix": user_means_for_matrix,
        "similarity": similarity_df,
    }


def predict_cf(user_id, tmdb_id, model):
    """
    Predict a rating using the user's mean and similar users'
    rating deviations.

    If collaborative information is insufficient, use a
    user/movie/global-mean fallback.
    """

    global_mean = model["global_mean"]
    user_means = model["user_means"]
    movie_means = model["movie_means"]
    rating_matrix = model["rating_matrix"]
    similarity_df = model["similarity"]

    user_mean = float(user_means.get(user_id, global_mean))
    movie_mean = float(movie_means.get(tmdb_id, global_mean))

    # User or movie not seen during training.
    if user_id not in rating_matrix.index:
        prediction = (
            0.5 * global_mean
            + 0.25 * user_mean
            + 0.25 * movie_mean
        )
        return float(np.clip(prediction, 1, 5))

    # Movie has no ratings in the training data.
    if tmdb_id not in rating_matrix.columns:
        prediction = (
            0.5 * global_mean
            + 0.25 * user_mean
            + 0.25 * movie_mean
        )
        return float(np.clip(prediction, 1, 5))

    similarities = similarity_df.loc[user_id].drop(user_id)

    # Keep only users who rated the target movie.
    movie_ratings = rating_matrix[tmdb_id].dropna()
    similarities = similarities.reindex(movie_ratings.index).dropna()

    # Use only positive similarities.
    similarities = similarities[similarities > 0]

    if similarities.empty:
        return float(np.clip(user_mean, 1, 5))

    # Select the most similar users.
    neighbors = similarities.nlargest(TOP_K_NEIGHBORS)

    neighbor_ratings = movie_ratings.reindex(neighbors.index)
    neighbor_means = user_means.reindex(neighbors.index)

    deviations = neighbor_ratings - neighbor_means

    denominator = float(neighbors.sum())

    if denominator <= 0:
        return float(np.clip(user_mean, 1, 5))

    prediction = user_mean + float(
        np.dot(neighbors.to_numpy(), deviations.to_numpy())
        / denominator
    )

    return float(np.clip(prediction, 1, 5))


# ============================================================
# 4. CONTENT-SIMILARITY FEATURES
# ============================================================

def calculate_content_similarity(user_id, target_tmdb_id, train_df):
    """
    Find the maximum TF-IDF cosine similarity between the target
    movie and movies the user rated 4 or higher in the training set.

    Only training interactions are used to determine liked movies.
    """

    target_index = movie_to_index.get(target_tmdb_id)

    if target_index is None:
        return 0.0

    liked_movies = train_df.loc[
        (train_df["user_id"] == user_id)
        & (train_df["rating"] >= 4),
        "tmdb_id"
    ].unique()

    liked_indices = [
        movie_to_index[movie_id]
        for movie_id in liked_movies
        if movie_id in movie_to_index
        and movie_id != target_tmdb_id
    ]

    if not liked_indices:
        return 0.0

    similarities = (
        tfidf_matrix[target_index]
        .dot(tfidf_matrix[liked_indices].T)
        .toarray()
        .ravel()
    )

    return float(similarities.max()) if len(similarities) else 0.0


# ============================================================
# 5. CREATE MODEL FEATURES
# ============================================================

def create_features(df, train_df, cf_model):
    """
    Create prediction features using only the supplied training data.
    """

    features = []

    for row in df.itertuples(index=False):
        user_id = int(row.user_id)
        tmdb_id = int(row.tmdb_id)

        collaborative_prediction = predict_cf(
            user_id,
            tmdb_id,
            cf_model
        )

        content_similarity = calculate_content_similarity(
            user_id,
            tmdb_id,
            train_df
        )

        user_mean = float(
            cf_model["user_means"].get(
                user_id,
                cf_model["global_mean"]
            )
        )

        movie_mean = float(
            cf_model["movie_means"].get(
                tmdb_id,
                cf_model["global_mean"]
            )
        )

        features.append([
            collaborative_prediction,
            content_similarity,
            user_mean,
            movie_mean,
        ])

    return pd.DataFrame(
        features,
        columns=[
            "collaborative_prediction",
            "content_similarity",
            "user_mean",
            "movie_mean",
        ]
    )


# ============================================================
# 6. SPLIT TRAINING DATA INTO TRAINING AND VALIDATION SETS
# ============================================================

model_train, validation = train_test_split(
    ratings,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nValidation experiment")
print("-" * 65)
print(f"Model-training interactions: {len(model_train):,}")
print(f"Validation interactions: {len(validation):,}")

validation_cf_model = build_cf_model(model_train)

print("\nCreating validation features...")

X_validation = create_features(
    validation,
    model_train,
    validation_cf_model
)

y_validation = validation["rating"].to_numpy()

# Learn the combination weights using validation data only.
hybrid_model = Ridge(alpha=10.0)

hybrid_model.fit(X_validation, y_validation)

print("\nLearned hybrid-model coefficients:")

for feature, coefficient in zip(
    X_validation.columns,
    hybrid_model.coef_
):
    print(f"{feature:30s}: {coefficient: .4f}")

print(f"{'Intercept':30s}: {hybrid_model.intercept_: .4f}")


# ============================================================
# 7. RETRAIN COLLABORATIVE FILTERING ON ALL TRAINING DATA
# ============================================================

print("\nTraining the final collaborative-filtering model...")

final_cf_model = build_cf_model(ratings)

print("Creating test features...")

X_test = create_features(
    test,
    ratings,
    final_cf_model
)

y_test = test["rating"].to_numpy()

cf_predictions = X_test["collaborative_prediction"].to_numpy()

hybrid_predictions = np.clip(
    hybrid_model.predict(X_test),
    1,
    5
)


# ============================================================
# 8. EVALUATE ON THE UNTOUCHED TEST SET
# ============================================================

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    return mae, rmse


baseline_predictions = np.full(
    len(test),
    final_cf_model["global_mean"]
)

baseline_mae, baseline_rmse = calculate_metrics(
    y_test,
    baseline_predictions
)

cf_mae, cf_rmse = calculate_metrics(
    y_test,
    cf_predictions
)

hybrid_mae, hybrid_rmse = calculate_metrics(
    y_test,
    hybrid_predictions
)

results = pd.DataFrame(
    {
        "Model": [
            "Global-average baseline",
            "Collaborative filtering",
            "Learned hybrid model",
        ],
        "MAE": [
            baseline_mae,
            cf_mae,
            hybrid_mae,
        ],
        "RMSE": [
            baseline_rmse,
            cf_rmse,
            hybrid_rmse,
        ],
    }
)

print("\n" + "=" * 65)
print("FINAL TEST-SET RESULTS")
print("=" * 65)

print(results.round(4).to_string(index=False))

print("\nLower MAE and RMSE are better.")

mae_improvement = cf_mae - hybrid_mae
rmse_improvement = cf_rmse - hybrid_rmse

print(f"\nHybrid MAE improvement over CF:  {mae_improvement:+.4f}")
print(f"Hybrid RMSE improvement over CF: {rmse_improvement:+.4f}")

if hybrid_mae < cf_mae and hybrid_rmse < cf_rmse:
    print("\nResult: The hybrid model improved both metrics.")
elif hybrid_mae < cf_mae or hybrid_rmse < cf_rmse:
    print("\nResult: The hybrid model improved one metric, but not both.")
else:
    print("\nResult: The hybrid model did not improve either metric.")


# ============================================================
# 9. SAVE TEST PREDICTIONS
# ============================================================

output = test.reset_index(drop=True).copy()

output["collaborative_prediction"] = cf_predictions
output["content_similarity"] = X_test["content_similarity"].to_numpy()
output["hybrid_prediction"] = hybrid_predictions

output_path = DATA_DIR / "learned_hybrid_predictions.csv"

output.to_csv(output_path, index=False)

print(f"\nTest predictions saved to: {output_path}")