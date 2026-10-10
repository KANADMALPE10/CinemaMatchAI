import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "ratings_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "ratings_test.csv"

print("=" * 60)
print("CINEMAMATCH AI - BASELINE MODEL EVALUATION")
print("=" * 60)

# Load training and testing datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"\nTraining ratings: {len(train):,}")
print(f"Testing ratings: {len(test):,}")

# Calculate average rating for every movie using training data only
movie_means = train.groupby("movie_id")["rating"].mean()

# Calculate overall average rating for the training dataset
global_mean = train["rating"].mean()

# Predict the average rating of each movie.
# If a movie was never rated in training, use the global average.
predictions = (
    test["movie_id"]
    .map(movie_means)
    .fillna(global_mean)
)

# Keep predictions within the possible rating range
predictions = predictions.clip(1, 5)

actual = test["rating"]

# Evaluate predictions
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

# Identify movies missing from the training data
unseen_mask = ~test["movie_id"].isin(movie_means.index)

print("\nEVALUATION RESULTS")
print("-" * 40)
print(f"Global average rating: {global_mean:.3f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\nTEST COVERAGE")
print("-" * 40)
print(f"Test ratings for known movies: {(~unseen_mask).sum():,}")
print(f"Test ratings for unseen movies: {unseen_mask.sum():,}")
print(
    f"Percentage of test ratings for unseen movies: "
    f"{unseen_mask.mean():.2%}"
)

print("\nBASELINE MODEL EVALUATION COMPLETED.")