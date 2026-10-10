import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 60)
print("CINEMATCH AI - EXPANDED RECOMMENDER")
print("=" * 60)


# --------------------------------------------------
# 2. Load catalog
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce",
)

df["release_year"] = df["release_date"].dt.year

for column in ["title", "genres", "overview", "content"]:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )

df = df.reset_index(drop=True)

print(f"\nMovies loaded: {len(df)}")


# --------------------------------------------------
# 3. Build TF-IDF vectors
# --------------------------------------------------

print("\nBuilding TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

tfidf_matrix = vectorizer.fit_transform(
    df["content"]
)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# --------------------------------------------------
# 4. Save reusable model components
# --------------------------------------------------

print("\nSaving model components...")

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib",
)

joblib.dump(
    tfidf_matrix,
    MODEL_DIR / "tfidf_matrix.joblib",
)

# Store movie metadata separately from the feature matrix.
movie_columns = [
    "tmdb_id",
    "title",
    "release_date",
    "release_year",
    "genres",
    "overview",
    "vote_average",
    "vote_count",
    "popularity",
    "poster_path",
]

joblib.dump(
    df[movie_columns],
    MODEL_DIR / "movie_metadata.joblib",
)

print("Vectorizer saved.")
print("TF-IDF matrix saved.")
print("Movie metadata saved.")


# --------------------------------------------------
# 5. Recommendation function
# --------------------------------------------------

def recommend_movies(
    movie_title,
    release_year=None,
    number_of_recommendations=10,
):
    """
    Recommend movies using content similarity.

    If multiple movies share a title, provide the
    release year to identify the intended movie.
    """

    matching = df[
        df["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if release_year is not None:
        matching = matching[
            matching["release_year"] == int(release_year)
        ]

    if matching.empty:
        print(
            f"\nNo matching movie found for "
            f"{movie_title!r}."
        )

        return pd.DataFrame()

    if len(matching) > 1:
        print(
            f"\nMultiple movies are titled {movie_title!r}."
        )

        print(
            matching[
                ["title", "release_year", "tmdb_id"]
            ].to_string(index=False)
        )

        print(
            "\nPlease call the function again with a release year."
        )

        return pd.DataFrame()

    movie_index = matching.index[0]

    # Compute similarities only for the selected movie.
    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix,
    ).ravel()

    # Exclude the selected movie.
    scores[movie_index] = -1

    ranked_indices = scores.argsort()[::-1]

    ranked_indices = ranked_indices[
        scores[ranked_indices] >= 0
    ][:number_of_recommendations]

    recommendations = df.iloc[
        ranked_indices
    ][movie_columns].copy()

    recommendations.insert(
        0,
        "similarity_score",
        scores[ranked_indices],
    )

    return recommendations.reset_index(drop=True)


# --------------------------------------------------
# 6. Test the expanded recommender
# --------------------------------------------------

test_movie = "Interstellar"

recommendations = recommend_movies(
    movie_title=test_movie,
    number_of_recommendations=10,
)

if not recommendations.empty:
    print(f"\nRecommendations for {test_movie}:")
    print("-" * 60)

    print(
        recommendations[
            [
                "similarity_score",
                "title",
                "release_year",
                "genres",
                "vote_average",
                "vote_count",
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 60)
print("MODEL BUILD COMPLETED")
print("=" * 60)