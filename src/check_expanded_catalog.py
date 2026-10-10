import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tmdb_catalog.csv"
)

df = pd.read_csv(DATA_PATH)

# Normalize titles for case-insensitive matching
df["title_normalized"] = (
    df["title"]
    .fillna("")
    .str.strip()
    .str.casefold()
)

test_movies = [
    "Interstellar",
    "The Dark Knight",
    "Inception",
    "The Shawshank Redemption",
    "The Godfather",
    "Titanic",
    "The Avengers",
    "Spider-Man: Brand New Day",
    "Resident Evil",
]

print("=" * 65)
print("CINEMAMATCH AI - EXPANDED CATALOG CHECK")
print("=" * 65)

for movie in test_movies:

    matches = df[
        df["title_normalized"] == movie.casefold()
    ]

    if matches.empty:
        print(f"\nNOT FOUND: {movie}")

    else:
        for _, row in matches.iterrows():

            print(f"\nFOUND: {row['title']}")
            print(f"Release date: {row['release_date']}")
            print(f"Genres: {row['genre_ids']}")
            print(f"Rating: {row['vote_average']}")
            print(f"Vote count: {row['vote_count']}")

print("\n" + "=" * 65)
print(f"Total catalog records: {len(df)}")
print(
    f"Unique movie IDs: {df['tmdb_id'].nunique()}"
)
print("=" * 65)