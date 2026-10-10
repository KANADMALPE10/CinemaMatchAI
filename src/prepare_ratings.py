import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RATINGS_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.data"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "ratings_train.csv"
TEST_PATH = OUTPUT_DIR / "ratings_test.csv"

print("=" * 60)
print("CINEMAMATCH AI - RATINGS TRAIN/TEST SPLIT")
print("=" * 60)

# Load MovieLens ratings
ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

print(f"\nTotal ratings loaded: {len(ratings):,}")

# Split ratings randomly into training and testing sets
train, test = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42,
)

# Save the datasets
train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print(f"Training ratings: {len(train):,}")
print(f"Testing ratings: {len(test):,}")

print("\nTRAINING DATA")
print("-" * 40)
print(f"Average rating: {train['rating'].mean():.3f}")
print(f"Unique users: {train['user_id'].nunique():,}")
print(f"Unique movies: {train['movie_id'].nunique():,}")

print("\nTESTING DATA")
print("-" * 40)
print(f"Average rating: {test['rating'].mean():.3f}")
print(f"Unique users: {test['user_id'].nunique():,}")
print(f"Unique movies: {test['movie_id'].nunique():,}")

print("\nSaved files:")
print(TRAIN_PATH)
print(TEST_PATH)

print("\nTrain/test split completed successfully.")