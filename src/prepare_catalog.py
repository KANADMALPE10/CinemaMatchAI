import json
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tmdb_catalog.csv"
)

GENRE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "genre_mapping.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_cleaned.csv"
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("=" * 60)
print("CINEMATCH AI - CATALOG PREPARATION")
print("=" * 60)

df = pd.read_csv(RAW_PATH)

with open(GENRE_PATH, "r", encoding="utf-8") as file:
    genre_mapping = json.load(file)

print(f"\nOriginal catalog: {len(df)} movies")


# --------------------------------------------------
# 3. Basic cleaning
# --------------------------------------------------

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce",
)

# Exclude records without an ID, title, or valid date.
df = df.dropna(
    subset=["tmdb_id", "title", "release_date"]
).copy()

df["title"] = df["title"].astype(str).str.strip()

df = df[df["title"] != ""].copy()

# Remove duplicate TMDB IDs.
df = df.drop_duplicates(
    subset=["tmdb_id"],
    keep="first",
).copy()


# --------------------------------------------------
# 4. Convert genre IDs into names
# --------------------------------------------------

def convert_genres(genre_ids):
    if pd.isna(genre_ids):
        return ""

    names = []

    for genre_id in str(genre_ids).split(","):
        genre_id = genre_id.strip()

        name = genre_mapping.get(genre_id)

        if name:
            names.append(name)

    return " ".join(names)


df["genres"] = df["genre_ids"].apply(convert_genres)


# --------------------------------------------------
# 5. Clean descriptions and posters
# --------------------------------------------------

for column in ["overview", "poster_path"]:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# --------------------------------------------------
# 6. Create content features
# --------------------------------------------------

df["content"] = (
    df["title"] + " "
    + df["genres"] + " "
    + df["genres"] + " "
    + df["overview"]
).str.strip()


# --------------------------------------------------
# 7. Add release year
# --------------------------------------------------

df["release_year"] = df["release_date"].dt.year


# --------------------------------------------------
# 8. Save prepared catalog
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nCatalog preparation completed!")

print(f"Movies retained: {len(df)}")
print(f"Unique TMDB IDs: {df['tmdb_id'].nunique()}")

print(f"\nSaved to:\n{OUTPUT_PATH}")

print("\nMovies per decade:")

print(
    (df["release_year"] // 10 * 10)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nMissing overviews:", (df["overview"] == "").sum())
print("Missing genres:", (df["genres"] == "").sum())

print("\nSample prepared movies:")

print(
    df[
        ["tmdb_id", "title", "release_year", "genres"]
    ]
    .head(10)
    .to_string(index=False)
)