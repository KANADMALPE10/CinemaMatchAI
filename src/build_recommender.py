import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_cleaned.csv"
)

print("=" * 60)
print("CINEMATCH AI - CONTENT-BASED RECOMMENDER")
print("=" * 60)


# --------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nMovies loaded: {len(df)}")


# --------------------------------------------------
# 3. Prepare movie content
# --------------------------------------------------

df["title"] = df["title"].fillna("").astype(str)
df["genres"] = df["genres"].fillna("").astype(str)
df["overview"] = df["overview"].fillna("").astype(str)

# Give genres extra weight by repeating them.
# This makes genre similarity more influential.
df["content"] = (
    df["title"] + " "
    + df["genres"] + " "
    + df["genres"] + " "
    + df["overview"]
).str.strip()

# Keep the original row index for matching recommendations.
df = df.reset_index(drop=True)


# --------------------------------------------------
# 4. Convert text into TF-IDF vectors
# --------------------------------------------------

print("\nConverting movie descriptions into TF-IDF vectors...")

tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=50000,

    ngram_range=(1, 2),

    min_df=2,

    sublinear_tf=True,

)

tfidf_matrix = tfidf.fit_transform(df["content"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# --------------------------------------------------
# 5. Calculate cosine similarity
# --------------------------------------------------

print("\nCalculating movie similarity...")

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False,
)

print(f"Similarity matrix shape: {similarity_matrix.shape}")


# --------------------------------------------------
# 6. Recommend similar movies
# --------------------------------------------------

def recommend_movies(movie_title, number_of_recommendations=10):
    """
    Recommend movies using TF-IDF and cosine similarity.

    The title search is case-insensitive.
    """

    matching_movies = df[
        df["title"].str.casefold() == movie_title.strip().casefold()
    ]

    if matching_movies.empty:
        print(f"\nMovie not found: {movie_title}")
        print("Try another title from the dataset.")
        return pd.DataFrame()

    # Use the first matching title if duplicates exist.
    movie_index = matching_movies.index[0]

    # Extract similarity scores for this movie.
    scores = similarity_matrix.getrow(movie_index).toarray().ravel()

    # Rank movies by similarity.
    ranked_indices = scores.argsort()[::-1]

    # Exclude the selected movie itself.
    ranked_indices = [
        index
        for index in ranked_indices
        if index != movie_index
    ]

    recommendations = df.iloc[
        ranked_indices[:number_of_recommendations]
    ][
        [
            "tmdb_id",
            "title",
            "release_date",
            "genres",
            "vote_average",
            "vote_count",
            "popularity",
            "poster_path",
        ]
    ].copy()

    recommendations.insert(
        0,
        "similarity_score",
        [float(scores[index]) for index in ranked_indices[:number_of_recommendations]],
    )

    return recommendations.reset_index(drop=True)


# --------------------------------------------------
# 7. Test the recommendation engine
# --------------------------------------------------

print("\nTesting the recommendation engine...")

test_movie = "Spider-Man: Brand New Day"

recommendations = recommend_movies(
    test_movie,
    number_of_recommendations=10,
)

if not recommendations.empty:
    print(f"\nRecommendations for: {test_movie}")
    print("-" * 60)

    print(
        recommendations.to_string(index=False)
    )

print("\nRecommendation engine test completed.")