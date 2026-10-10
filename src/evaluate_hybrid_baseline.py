from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "ratings_train_hybrid.csv"
TEST_PATH = DATA_DIR / "ratings_test_hybrid.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Learn all baseline statistics from training data only.
global_mean = train["rating"].mean()
movie_means = train.groupby("tmdb_id")["rating"].mean()

# Predict using movie averages, with global fallback for unseen movies.
predictions = test["tmdb_id"].map(movie_means)
fallback_count = predictions.isna().sum()

predictions = predictions.fillna(global_mean).clip(1, 5)

actual = test["rating"]

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("=" * 60)
print("CINEMAMATCH AI - CORRECTED BASELINE EVALUATION")
print("=" * 60)

print(f"Training ratings: {len(train):,}")
print(f"Test ratings: {len(test):,}")
print(f"Global mean rating: {global_mean:.4f}")
print(f"Test ratings using fallback: {fallback_count:,}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
