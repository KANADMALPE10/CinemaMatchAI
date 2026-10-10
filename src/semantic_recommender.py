from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "catalog_cleaned.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_MOVIE_ID = 27205  # Inception
RECOMMENDATION_LIMIT = 12


# --------------------------------------------------
# LOAD MOVIE CATALOG
# --------------------------------------------------

def load_catalog():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Movie catalog not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "tmdb_id",
        "title",
        "overview",
        "genres",
        "release_year",
        "vote_average",
        "vote_count",
        "poster_path",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in catalog: {missing_columns}"
        )

    df = df.copy()

    df["tmdb_id"] = pd.to_numeric(
        df["tmdb_id"], errors="coerce"
    )

    df = df.dropna(subset=["tmdb_id", "title"])

    df["tmdb_id"] = df["tmdb_id"].astype(int)

    df = df.drop_duplicates(
        subset=["tmdb_id"]
    ).reset_index(drop=True)

    for column in [
        "overview",
        "genres",
        "release_year",
        "vote_average",
        "vote_count",
        "poster_path",
    ]:
        df[column] = df[column].fillna("")

    df["overview"] = df["overview"].astype(str)
    df["genres"] = df["genres"].astype(str)

    df["vote_average"] = pd.to_numeric(
        df["vote_average"], errors="coerce"
    ).fillna(0.0)

    df["vote_count"] = pd.to_numeric(
        df["vote_count"], errors="coerce"
    ).fillna(0)

    print(f"Loaded {len(df):,} movies.")

    return df


# --------------------------------------------------
# BUILD MOVIE TEXT
# --------------------------------------------------

def build_movie_text(row):
    title = str(row["title"]).strip()
    overview = str(row["overview"]).strip()
    genres = str(row["genres"]).strip()

    # Give the plot the most space.
    return (
        f"Movie title: {title}. "
        f"Genres: {genres}. "
        f"Story: {overview}"
    )


# --------------------------------------------------
# SEMANTIC RECOMMENDER
# --------------------------------------------------

class SemanticMovieRecommender:

    def __init__(self):
        self.df = load_catalog()

        print("\nLoading semantic language model...")
        self.model = SentenceTransformer(MODEL_NAME)

                # Cache embeddings to avoid recalculating them.
        cache_dir = PROJECT_ROOT / "models" / "semantic"
        cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = cache_dir / "movie_embeddings.npy"
        ids_path = cache_dir / "movie_ids.npy"

        current_ids = self.df["tmdb_id"].to_numpy(dtype=np.int64)

        cache_is_valid = False

        if embeddings_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path)

            if np.array_equal(cached_ids, current_ids):
                cached_embeddings = np.load(embeddings_path)

                if (
                    cached_embeddings.shape[0] == len(self.df)
                    and cached_embeddings.ndim == 2
                ):
                    self.embeddings = cached_embeddings
                    cache_is_valid = True
                    print("Loaded cached movie embeddings.")

        if not cache_is_valid:
            print("Preparing movie descriptions...")

            movie_texts = [
                build_movie_text(row)
                for _, row in self.df.iterrows()
            ]

            print(
                f"Generating semantic embeddings for "
                f"{len(movie_texts):,} movies..."
            )

            self.embeddings = self.model.encode(
                movie_texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            np.save(embeddings_path, self.embeddings)
            np.save(ids_path, current_ids)

            print("Saved movie embeddings to disk.")

        print("Semantic embeddings are ready.")

    def recommend(self, movie_id, limit=12):

        matching_rows = self.df.index[
            self.df["tmdb_id"] == int(movie_id)
        ].tolist()

        if not matching_rows:
            raise ValueError(
                f"Movie ID {movie_id} was not found."
            )

        movie_index = matching_rows[0]

        selected_movie = self.df.iloc[movie_index]

        # Calculate similarity against every movie.
        similarities = self.embeddings @ self.embeddings[
            movie_index
        ]

        candidate_scores = []

        for index, semantic_score in enumerate(similarities):

            # Do not recommend the selected movie itself.
            if index == movie_index:
                continue

            candidate = self.df.iloc[index]

            # Slightly favor movies with good ratings.
            # Ratings are not allowed to overpower story similarity.
            rating = float(candidate["vote_average"])
            vote_count = float(candidate["vote_count"])

            rating_confidence = min(
                vote_count / 500.0,
                1.0
            )

            quality_score = (
                (rating / 10.0) * rating_confidence
            )

            final_score = (
                0.88 * float(semantic_score)
                + 0.12 * quality_score
            )

            candidate_scores.append(
                (
                    index,
                    float(semantic_score),
                    quality_score,
                    final_score,
                )
            )

        candidate_scores.sort(
            key=lambda item: item[3],
            reverse=True,
        )

        results = []

        for (
            index,
            semantic_score,
            quality_score,
            final_score,
        ) in candidate_scores[:limit]:

            movie = self.df.iloc[index]

            results.append({
                "tmdb_id": int(movie["tmdb_id"]),
                "title": str(movie["title"]),
                "genres": str(movie["genres"]),
                "year": str(movie["release_year"]),
                "rating": round(
                    float(movie["vote_average"]), 2
                ),
                "semantic_similarity": round(
                    semantic_score, 4
                ),
                "quality_score": round(
                    quality_score, 4
                ),
                "final_score": round(
                    final_score, 4
                ),
                "poster_path": str(movie["poster_path"]),
                "overview": str(movie["overview"]),
            })

        return selected_movie, results


# --------------------------------------------------
# TEST THE RECOMMENDER
# --------------------------------------------------

def main():

    recommender = SemanticMovieRecommender()

    selected_movie, recommendations = recommender.recommend(
        TEST_MOVIE_ID,
        limit=RECOMMENDATION_LIMIT,
    )

    print("\n" + "=" * 65)
    print(
        f"RECOMMENDATIONS FOR: {selected_movie['title']}"
    )
    print("=" * 65)

    for position, movie in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"\n{position}. {movie['title']}"
            f"\n   Genres: {movie['genres']}"
            f"\n   Rating: {movie['rating']}"
            f"\n   Semantic similarity: "
            f"{movie['semantic_similarity']:.3f}"
            f"\n   Final score: {movie['final_score']:.3f}"
            f"\n   Overview: "
            f"{movie['overview'][:180]}"
        )

    print("\n" + "=" * 65)
    print("SEMANTIC RECOMMENDER TEST COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()