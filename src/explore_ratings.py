import pandas as pd

from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "ratings"
    / "ml-100k"
    / "u.data"
)

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "ratings"
    / "ml-100k"
    / "u.item"
)


# --------------------------------------------------
# 2. Load ratings
# --------------------------------------------------

print("=" * 60)
print("CINEMATCH AI - MOVIELENS RATINGS EXPLORATION")
print("=" * 60)

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
    ],
)

print(f"\nTotal ratings: {len(ratings):,}")


# --------------------------------------------------
# 3. Load movie metadata
# --------------------------------------------------

movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1, 2],
    names=[
        "movie_id",
        "title",
        "release_date",
    ],
)

print(f"Movie metadata records: {len(movies):,}")


# --------------------------------------------------
# 4. Basic dataset statistics
# --------------------------------------------------

print("\nDATASET STATISTICS")
print("-" * 40)

print(f"Unique users: {ratings['user_id'].nunique():,}")
print(f"Unique rated movies: {ratings['movie_id'].nunique():,}")
print(f"Unique movie records: {movies['movie_id'].nunique():,}")

print(f"Average rating: {ratings['rating'].mean():.3f}")
print(f"Median rating: {ratings['rating'].median():.1f}")

print(
    f"Rating range: "
    f"{ratings['rating'].min()} to "
    f"{ratings['rating'].max()}"
)


# --------------------------------------------------
# 5. Rating distribution
# --------------------------------------------------

print("\nRATING DISTRIBUTION")
print("-" * 40)

print(
    ratings["rating"]
    .value_counts()
    .sort_index()
    .to_string()
)


# --------------------------------------------------
# 6. Most-rated movies
# --------------------------------------------------

rating_counts = (
    ratings
    .groupby("movie_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
    .reset_index()
)

top_movies = (
    rating_counts
    .merge(
        movies,
        on="movie_id",
        how="left",
    )
    .sort_values(
        "rating_count",
        ascending=False,
    )
    .head(10)
)

print("\nTOP 10 MOST-RATED MOVIES")
print("-" * 40)

print(
    top_movies[
        [
            "title",
            "rating_count",
            "average_rating",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 7. Rating matrix sparsity
# --------------------------------------------------

num_users = ratings["user_id"].nunique()
num_movies = ratings["movie_id"].nunique()

possible_ratings = num_users * num_movies

density = len(ratings) / possible_ratings

print("\nRATING MATRIX")
print("-" * 40)

print(f"Users: {num_users:,}")
print(f"Rated movies: {num_movies:,}")
print(f"Possible user-movie pairs: {possible_ratings:,}")
print(f"Observed ratings: {len(ratings):,}")
print(f"Matrix density: {density:.4%}")
print(f"Matrix sparsity: {(1 - density):.4%}")


# --------------------------------------------------
# 8. Sample ratings
# --------------------------------------------------

print("\nFIRST 10 RATINGS")
print("-" * 40)

sample = ratings.merge(
    movies[["movie_id", "title"]],
    on="movie_id",
    how="left",
)

print(
    sample[
        [
            "user_id",
            "movie_id",
            "title",
            "rating",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nRatings exploration completed.")