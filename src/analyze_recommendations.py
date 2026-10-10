import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ratings_train.csv"
)

MOVIES_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.item"
)

train = pd.read_csv(TRAIN_PATH)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

# Calculate rating statistics for each movie.
movie_stats = (
    train.groupby("movie_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
    .reset_index()
)

movie_stats = movie_stats.merge(
    movies,
    on="movie_id",
    how="left",
)

# Calculate confidence-adjusted scores.
global_mean = train["rating"].mean()

# Use 20 ratings as the initial minimum-rating threshold.
m = 20

movie_stats["bayesian_score"] = (
    (movie_stats["rating_count"] /
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m /
     (movie_stats["rating_count"] + m))
    * global_mean
)

print("=" * 65)
print("CINEMAMATCH AI - MOVIE RATING RELIABILITY")
print("=" * 65)

print(f"\nOverall average rating: {global_mean:.3f}")

print("\nMOST-RATED MOVIES")
print("-" * 65)

popular = movie_stats.sort_values(
    "rating_count",
    ascending=False,
).head(10)

print(
    popular[
        ["title", "rating_count", "average_rating", "bayesian_score"]
    ].to_string(
        index=False,
        formatters={
            "average_rating": "{:.2f}".format,
            "bayesian_score": "{:.2f}".format,
        },
    )
)

print("\nHIGHEST CONFIDENCE-ADJUSTED SCORES")
print("-" * 65)

reliable = movie_stats.sort_values(
    ["bayesian_score", "rating_count"],
    ascending=[False, False],
).head(10)

print(
    reliable[
        ["title", "rating_count", "average_rating", "bayesian_score"]
    ].to_string(
        index=False,
        formatters={
            "average_rating": "{:.2f}".format,
            "bayesian_score": "{:.2f}".format,
        },
    )
)

print("\nRATING RELIABILITY ANALYSIS COMPLETED.")