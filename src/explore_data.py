import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movies_2026.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Dataset overview
print("=" * 60)
print("CINEMATCH AI - DATASET OVERVIEW")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 movies:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate movie IDs:")
print(df["tmdb_id"].duplicated().sum())

print("\nRelease date range:")
print(df["release_date"].min())
print(df["release_date"].max())

print("\nTop 10 most popular movies:")
print(
    df[["title", "release_date", "popularity"]]
    .sort_values("popularity", ascending=False)
    .head(10)
    .to_string(index=False)
)