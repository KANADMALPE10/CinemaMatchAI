import json
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tmdb_movies_2026.csv"
)

GENRE_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "genre_mapping.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_cleaned.csv"
)


# --------------------------------------------------
# 2. Load the dataset and genre mapping
# --------------------------------------------------

print("=" * 60)
print("CINEMATCH AI - DATA CLEANING")
print("=" * 60)

df = pd.read_csv(RAW_DATA_PATH)

with open(GENRE_MAPPING_PATH, "r", encoding="utf-8") as file:
    genre_mapping = json.load(file)

print(f"\nOriginal dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Convert release dates
# --------------------------------------------------

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce",
)

invalid_dates = df["release_date"].isna().sum()

print(f"Invalid release dates: {invalid_dates}")

df = df.dropna(subset=["release_date"])


# --------------------------------------------------
# 4. Filter the intended release-date range
# --------------------------------------------------

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-10-09")

df = df[
    df["release_date"].between(
        START_DATE,
        END_DATE,
        inclusive="both",
    )
].copy()

print(f"Records within the date range: {len(df)}")


# --------------------------------------------------
# 5. Remove duplicate movie IDs
# --------------------------------------------------

df = df.drop_duplicates(
    subset=["tmdb_id"],
    keep="first",
).copy()


# --------------------------------------------------
# 6. Convert genre IDs into genre names
# --------------------------------------------------

def convert_genres(genre_ids):
    if pd.isna(genre_ids):
        return ""

    genre_names = []

    for genre_id in str(genre_ids).split(","):
        genre_id = genre_id.strip()

        genre_name = genre_mapping.get(genre_id)

        if genre_name:
            genre_names.append(genre_name)

    return " ".join(genre_names)


df["genres"] = df["genre_ids"].apply(convert_genres)


# --------------------------------------------------
# 7. Clean movie overviews
# --------------------------------------------------

df["overview"] = (
    df["overview"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# 8. Clean poster paths
# --------------------------------------------------

df["poster_path"] = (
    df["poster_path"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# 9. Create a combined content feature
# --------------------------------------------------

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["genres"]
    + " "
    + df["overview"]
).str.strip()


# --------------------------------------------------
# 10. Save the cleaned dataset
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"\nFinal dataset shape: {df.shape}")

print(f"Dataset saved to:\n{OUTPUT_PATH}")

print("\nMissing values after cleaning:")

print(
    df[
        ["title", "release_date", "overview", "genres", "poster_path"]
    ]
    .isna()
    .sum()
)

print("\nMovies with missing overviews:", (df["overview"] == "").sum())
print("Movies with missing genres:", (df["genres"] == "").sum())

print("\nFirst 10 cleaned movies:")

print(
    df[
        ["title", "release_date", "genres", "overview"]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTop 10 movies by popularity:")

print(
    df[
        ["title", "genres", "popularity"]
    ]
    .sort_values("popularity", ascending=False)
    .head(10)
    .to_string(index=False)
)