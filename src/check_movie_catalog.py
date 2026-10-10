import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

test_movies = [
    "Interstellar",
    "The Dark Knight",
    "Inception",
    "Spider-Man: Brand New Day",
    "Resident Evil",
]

print("=" * 60)
print("CINEMATCH AI - MOVIE CATALOG CHECK")
print("=" * 60)

for movie in test_movies:
    matches = df[
        df["title"].str.casefold() == movie.casefold()
    ]

    if matches.empty:
        print(f"\n❌ {movie}: Not found")
    else:
        for _, row in matches.iterrows():
            print(
                f"\n✅ {row['title']} "
                f"| Release: {row['release_date']} "
                f"| Genres: {row['genres']}"
            )

print("\n" + "=" * 60)
print(f"Total movies in catalog: {len(df)}")
print("=" * 60)