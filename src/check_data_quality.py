import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movies_2026.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CINEMATCH AI - DATA QUALITY CHECK")
print("=" * 60)

# 1. Release year distribution
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

print("\n1. MOVIES BY RELEASE YEAR")
print("-" * 40)

print(
    df["release_date"]
    .dt.year
    .value_counts()
    .sort_index()
    .to_string()
)

# 2. Movies released in 2026
movies_2026 = df[
    df["release_date"].dt.year == 2026
]

print("\n2. 2026 MOVIES")
print("-" * 40)
print(f"Movies released in 2026: {len(movies_2026)}")

# 3. Dates outside the intended range
outside_range = df[
    (df["release_date"] < "2026-01-01")
    | (df["release_date"] > "2026-10-09")
]

print("\n3. MOVIES OUTSIDE THE INTENDED DATE RANGE")
print("-" * 40)
print(f"Records outside the date range: {len(outside_range)}")

print(
    outside_range[
        ["title", "release_date"]
    ]
    .head(10)
    .to_string(index=False)
)

# 4. Movies with no overview
print("\n4. MISSING OVERVIEWS")
print("-" * 40)

print(
    f"Missing overviews: {df['overview'].isna().sum()}"
)

# 5. Movies with no genres
print("\n5. MISSING GENRES")
print("-" * 40)

print(
    f"Missing genre information: {df['genre_ids'].isna().sum()}"
)

# 6. Rating statistics
print("\n6. RATING STATISTICS")
print("-" * 40)

print(df["vote_average"].describe())

# 7. Vote-count statistics
print("\n7. VOTE COUNT STATISTICS")
print("-" * 40)

print(df["vote_count"].describe())

print("\nData-quality check completed.")