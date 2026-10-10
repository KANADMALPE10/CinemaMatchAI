from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RATINGS_PATH = ROOT / "data/ratings/ml-100k/u.data"

MAPPING_PATH = (
    ROOT / "data/processed/movie_catalog_mapping_approved.csv"
)

OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "ratings_train_hybrid.csv"
TEST_PATH = OUTPUT_DIR / "ratings_test_hybrid.csv"


# ============================================================
# 2. LOAD ORIGINAL DATA
# ============================================================

print("=" * 65)
print("CINEMAMATCH AI - HYBRID DATA PREPARATION")
print("=" * 65)

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

mapping = pd.read_csv(MAPPING_PATH)

print(f"\nOriginal ratings: {len(ratings):,}")

# Retain only approved mappings with valid TMDB IDs.
mapping = mapping.dropna(subset=["tmdb_id"]).copy()

mapping["movie_id"] = mapping["movie_id"].astype(int)
mapping["tmdb_id"] = mapping["tmdb_id"].astype(int)

# One MovieLens ID must map to exactly one TMDB ID.
if mapping["movie_id"].duplicated().any():
    raise ValueError(
        "Duplicate MovieLens IDs found in the mapping."
    )

movie_id_to_tmdb = dict(
    zip(mapping["movie_id"], mapping["tmdb_id"])
)


# ============================================================
# 3. MAP AND CONSOLIDATE RATINGS
# ============================================================

ratings["tmdb_id"] = ratings["movie_id"].map(movie_id_to_tmdb)

unmatched_count = ratings["tmdb_id"].isna().sum()

ratings = ratings.dropna(subset=["tmdb_id"]).copy()
ratings["tmdb_id"] = ratings["tmdb_id"].astype(int)

print(f"Ratings without a TMDB match: {unmatched_count:,}")
print(f"Ratings with a TMDB match: {len(ratings):,}")

# Consolidate duplicate movie entries for each user BEFORE splitting.
# Preserve the average rating and latest timestamp for each pair.
consolidated = (
    ratings.groupby(
        ["user_id", "tmdb_id"],
        as_index=False,
    )
    .agg(
        rating=("rating", "mean"),
        timestamp=("timestamp", "max"),
    )
)

print(f"Consolidated user-movie ratings: {len(consolidated):,}")
print(f"Unique users: {consolidated['user_id'].nunique():,}")
print(f"Unique TMDB movies: {consolidated['tmdb_id'].nunique():,}")


# ============================================================
# 4. SPLIT AFTER CONSOLIDATION
# ============================================================

train, test = train_test_split(
    consolidated,
    test_size=0.20,
    random_state=42,
)

# Save separate files; leave all existing files unchanged.
train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print("\nTRAINING DATA")
print("-" * 45)
print(f"Ratings: {len(train):,}")
print(f"Users: {train['user_id'].nunique():,}")
print(f"Movies: {train['tmdb_id'].nunique():,}")
print(f"Mean rating: {train['rating'].mean():.3f}")

print("\nTEST DATA")
print("-" * 45)
print(f"Ratings: {len(test):,}")
print(f"Users: {test['user_id'].nunique():,}")
print(f"Movies: {test['tmdb_id'].nunique():,}")
print(f"Mean rating: {test['rating'].mean():.3f}")

print("\nSaved files:")
print(TRAIN_PATH)
print(TEST_PATH)

print("\nHybrid rating preparation completed successfully.")