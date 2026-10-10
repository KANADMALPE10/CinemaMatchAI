from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "ratings_train_consolidated.csv")
test = pd.read_csv(DATA_DIR / "ratings_test_consolidated.csv")

# Calculate movie averages using training data only.
global_mean = train["rating"].mean()
movie_means = train.groupby("tmdb_id")["rating"].mean()

# Predict each test rating.
predictions = test["tmdb_id"].map(movie_means)
predictions = predictions.fillna(global_mean)

# Ratings must remain within the valid 1–5 range.
predictions = predictions.clip(1, 5)

mae = mean_absolute_error(test["rating"], predictions)
rmse = np.sqrt(mean_squared_error(test["rating"], predictions))

print("=" * 55)
print("CINEMAMATCH AI - CONSOLIDATED BASELINE EVALUATION")
print("=" * 55)

print(f"Training ratings: {len(train):,}")
print(f"Test ratings: {len(test):,}")
print(f"Global training mean: {global_mean:.4f}")
print(f"Test ratings using global fallback: {test['tmdb_id'].isin(set(test.loc[~test['tmdb_id'].isin(movie_means.index), 'tmdb_id'])).sum() if False else test['tmdb_id'].map(movie_means).isna().sum():,}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
