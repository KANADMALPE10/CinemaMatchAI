from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "ratings_train.csv"
TEST_PATH = DATA_DIR / "ratings_test.csv"

MAPPING_PATH = DATA_DIR / "movie_catalog_mapping_approved.csv"

TRAIN_OUTPUT = DATA_DIR / "ratings_train_consolidated.csv"
TEST_OUTPUT = DATA_DIR / "ratings_test_consolidated.csv"

# Load original ratings and the approved movie mapping.
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
mapping = pd.read_csv(MAPPING_PATH)

# Build MovieLens ID -> TMDB ID mapping.
matched = mapping.dropna(subset=["tmdb_id"]).copy()

matched["movie_id"] = matched["movie_id"].astype(int)
matched["tmdb_id"] = matched["tmdb_id"].astype(int)

id_mapping = dict(zip(matched["movie_id"], matched["tmdb_id"]))


def consolidate_ratings(ratings):
    result = ratings.copy()

    # Retain only ratings for movies that have a TMDB match.
    result["tmdb_id"] = result["movie_id"].map(id_mapping)
    result = result.dropna(subset=["tmdb_id"]).copy()

    result["tmdb_id"] = result["tmdb_id"].astype(int)

    # Keep each user's separate rating event.
    # If a user rated duplicate entries for the same TMDB movie,
    # combine those ratings using their mean.
    result = (
        result.groupby(["user_id", "tmdb_id"], as_index=False)
        .agg(
            rating=("rating", "mean"),
            timestamp=("timestamp", "max"),
        )
    )

    return result


train_consolidated = consolidate_ratings(train)
test_consolidated = consolidate_ratings(test)

train_consolidated.to_csv(TRAIN_OUTPUT, index=False)
test_consolidated.to_csv(TEST_OUTPUT, index=False)

print("=" * 60)
print("CINEMAMATCH AI - RATINGS CONSOLIDATION")
print("=" * 60)

print(f"\nOriginal training ratings: {len(train)}")
print(f"Consolidated training ratings: {len(train_consolidated)}")

print(f"\nOriginal test ratings: {len(test)}")
print(f"Consolidated test ratings: {len(test_consolidated)}")

print(f"\nUnique TMDB movies in training: {train_consolidated['tmdb_id'].nunique()}")
print(f"Unique users in training: {train_consolidated['user_id'].nunique()}")

print("\nSaved files:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)
