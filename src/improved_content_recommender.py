from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "catalog_cleaned.csv"
MODEL_DIR = ROOT / "models"

TFIDF_PATH = MODEL_DIR / "tfidf_matrix.joblib"
METADATA_PATH = MODEL_DIR / "movie_metadata.joblib"

# Similarity weights. These are starting values, not validated
# optimal weights.
TITLE_WEIGHT = 0.10
OVERVIEW_WEIGHT = 0.65
GENRE_WEIGHT = 0.10
EXISTING_WEIGHT = 0.15

QUALITY_WEIGHT = 0.04
FRANCHISE_BOOST = 0.10

TEST_MOVIE_ID = 27205  # Inception
RECOMMENDATION_LIMIT = 12


# ============================================================
# FRANCHISE RULES
# ============================================================

FRANCHISE_RULES = {
    "avatar": [
        "avatar: the way of water",
        "avatar: fire and ash",
    ],
    "the matrix": [
        "the matrix reloaded",
        "the matrix revolutions",
        "the matrix resurrections",
    ],
    "the lord of the rings": [
        "the lord of the rings: the two towers",
        "the lord of the rings: the return of the king",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def normalize_title(title):
    title = str(title).strip().casefold()
    title = re.sub(r"[^\w\s]", "", title)
    return re.sub(r"\s+", " ", title)


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


# ============================================================
# RECOMMENDER
# ============================================================

class ImprovedContentRecommender:

    def __init__(self):

        print("Loading movie catalog...")

        self.movies = pd.read_csv(DATA_PATH)

        required_columns = {
            "tmdb_id",
            "title",
            "overview",
            "genres",
        }

        missing = required_columns - set(self.movies.columns)

        if missing:
            raise ValueError(
                f"Missing catalog columns: {sorted(missing)}"
            )

        self.movies["tmdb_id"] = pd.to_numeric(
            self.movies["tmdb_id"],
            errors="coerce",
        )

        self.movies = self.movies.dropna(
            subset=["tmdb_id", "title"]
        ).copy()

        self.movies["tmdb_id"] = (
            self.movies["tmdb_id"].astype(int)
        )

        for column in ["title", "overview", "genres"]:
            self.movies[column] = (
                self.movies[column].fillna("").astype(str)
            )

        self.movies = (
            self.movies.drop_duplicates("tmdb_id")
            .reset_index(drop=True)
        )

        self.movies["normalized_title"] = (
            self.movies["title"].map(normalize_title)
        )

        for column in ["vote_average", "vote_count", "popularity"]:
            if column not in self.movies.columns:
                self.movies[column] = 0

            self.movies[column] = pd.to_numeric(
                self.movies[column],
                errors="coerce",
            ).fillna(0)

        print(f"Loaded {len(self.movies):,} movies.")

        # ----------------------------------------------------
        # TITLE MODEL
        # ----------------------------------------------------

        print("Building title similarity model...")

        self.title_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.title_matrix = self.title_vectorizer.fit_transform(
            self.movies["title"]
        )

        # ----------------------------------------------------
        # OVERVIEW MODEL
        # ----------------------------------------------------

        print("Building story similarity model...")

        self.overview_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        self.overview_matrix = self.overview_vectorizer.fit_transform(
            self.movies["overview"]
        )

        # ----------------------------------------------------
        # GENRE MODEL
        # ----------------------------------------------------

        print("Building genre similarity model...")

        self.genre_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=True,
        )

        self.genre_matrix = self.genre_vectorizer.fit_transform(
            self.movies["genres"]
        )

        # ----------------------------------------------------
        # EXISTING CONTENT MODEL
        # ----------------------------------------------------

        self.old_tfidf = None
        self.old_index = {}

        if TFIDF_PATH.exists() and METADATA_PATH.exists():

            try:
                print("Loading existing content model...")

                old_metadata = joblib.load(METADATA_PATH)
                old_tfidf = joblib.load(TFIDF_PATH)

                old_metadata["tmdb_id"] = pd.to_numeric(
                    old_metadata["tmdb_id"],
                    errors="coerce",
                )

                old_metadata = old_metadata.dropna(
                    subset=["tmdb_id"]
                ).copy()

                old_metadata["tmdb_id"] = (
                    old_metadata["tmdb_id"].astype(int)
                )

                old_ids = old_metadata["tmdb_id"].tolist()

                if (
                    len(old_ids) == len(set(old_ids))
                    and old_tfidf.shape[0] == len(old_ids)
                ):
                    self.old_tfidf = old_tfidf

                    self.old_index = {
                        movie_id: index
                        for index, movie_id in enumerate(old_ids)
                    }

                    print("Existing content model loaded.")

            except Exception as exc:
                print(
                    "Existing content model unavailable: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # INDEX
        # ----------------------------------------------------

        self.new_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movies["tmdb_id"])
        }

        # ----------------------------------------------------
        # QUALITY SCORE
        # ----------------------------------------------------

        ratings = self.movies["vote_average"].to_numpy(dtype=float)
        votes = self.movies["vote_count"].to_numpy(dtype=float)

        positive_votes = votes[votes > 0]

        prior_votes = (
            float(np.median(positive_votes))
            if len(positive_votes)
            else 1.0
        )

        global_rating = (
            float(
                np.average(
                    ratings,
                    weights=np.maximum(votes, 1),
                )
            )
            if len(ratings)
            else 0.0
        )

        global_rating = np.clip(global_rating, 0, 10)

        weighted_rating = (
            votes / (votes + prior_votes)
        ) * ratings + (
            prior_votes / (votes + prior_votes)
        ) * global_rating

        self.quality_scores = np.clip(
            weighted_rating / 10.0,
            0,
            1,
        )

        print("Improved recommender is ready.")

    # ========================================================
    # FRANCHISE BONUS
    # ========================================================

    def _get_franchise_boost(self, target_title):

        boosts = np.zeros(len(self.movies), dtype=float)

        target_key = normalize_title(target_title)

        for franchise, related_titles in FRANCHISE_RULES.items():

            franchise_key = normalize_title(franchise)

            if target_key == franchise_key:

                related_keys = {
                    normalize_title(title)
                    for title in related_titles
                }

                matching = (
                    self.movies["normalized_title"]
                    .isin(related_keys)
                    .to_numpy()
                )

                boosts[matching] = FRANCHISE_BOOST

        return boosts

    # ========================================================
    # RECOMMEND
    # ========================================================

    def recommend(self, tmdb_id, limit=12):

        tmdb_id = int(tmdb_id)

        if tmdb_id not in self.new_index:
            raise ValueError(
                f"TMDB ID {tmdb_id} was not found."
            )

        target_index = self.new_index[tmdb_id]
        target = self.movies.iloc[target_index]

        # Title similarity.
        title_scores = cosine_similarity(
            self.title_matrix[target_index],
            self.title_matrix,
        ).ravel()

        # Story similarity.
        overview_scores = cosine_similarity(
            self.overview_matrix[target_index],
            self.overview_matrix,
        ).ravel()

        # Genre similarity.
        genre_scores = cosine_similarity(
            self.genre_matrix[target_index],
            self.genre_matrix,
        ).ravel()

        # Existing content similarity.
        existing_scores = np.zeros(
            len(self.movies),
            dtype=float,
        )

        old_index = self.old_index.get(tmdb_id)

        if self.old_tfidf is not None and old_index is not None:

            scores = (
                self.old_tfidf[old_index]
                .dot(self.old_tfidf.T)
            )

            if hasattr(scores, "toarray"):
                scores = scores.toarray()

            scores = np.asarray(scores).ravel()

            for index, movie_id in enumerate(self.movies["tmdb_id"]):

                previous_index = self.old_index.get(int(movie_id))

                if previous_index is not None:
                    existing_scores[index] = scores[previous_index]

        # Combine signals.
        combined = (
            TITLE_WEIGHT * title_scores
            + OVERVIEW_WEIGHT * overview_scores
            + GENRE_WEIGHT * genre_scores
            + EXISTING_WEIGHT * existing_scores
        )

        # Small quality adjustment.
        combined += QUALITY_WEIGHT * self.quality_scores

        # Franchise adjustment.
        franchise_boost = self._get_franchise_boost(
            target["title"]
        )

        combined += franchise_boost

        # Exclude selected movie and duplicate titles.
        target_key = normalize_title(target["title"])

        combined[
            self.movies["normalized_title"].to_numpy() == target_key
        ] = -1

        ranked_indices = np.argsort(combined)[::-1]

        results = []
        seen = {target_key}

        for index in ranked_indices:

            if combined[index] < 0:
                continue

            movie = self.movies.iloc[index]

            title = str(movie["title"]).strip()
            title_key = normalize_title(title)

            if not title or title_key in seen:
                continue

            seen.add(title_key)

            year = movie.get("release_year")

            if pd.isna(year):
                year = None
            else:
                year = int(year)

            poster = movie.get("poster_path")

            if pd.isna(poster):
                poster = None

            results.append({
                "tmdb_id": int(movie["tmdb_id"]),
                "title": title,
                "genres": str(movie["genres"]),
                "year": year,
                "rating": safe_float(movie["vote_average"]),
                "overview_similarity": round(
                    float(overview_scores[index]), 4
                ),
                "genre_similarity": round(
                    float(genre_scores[index]), 4
                ),
                "existing_similarity": round(
                    float(existing_scores[index]), 4
                ),
                "franchise_boost": round(
                    float(franchise_boost[index]), 4
                ),
                "similarity": round(
                    float(combined[index]), 4
                ),
                "poster_path": poster,
            })

            if len(results) >= limit:
                break

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    recommender = ImprovedContentRecommender()

    matches = recommender.movies[
        recommender.movies["tmdb_id"] == TEST_MOVIE_ID
    ]

    if matches.empty:
        raise ValueError(
            f"Test movie ID {TEST_MOVIE_ID} not found."
        )

    title = matches.iloc[0]["title"]

    recommendations = recommender.recommend(
        TEST_MOVIE_ID,
        RECOMMENDATION_LIMIT,
    )

    print(f"\nRecommendations for {title}")
    print("=" * 110)

    for rank, movie in enumerate(recommendations, 1):

        print(
            f"{rank:2}. {movie['title']:<42} "
            f"| Story: {movie['overview_similarity']:.3f} "
            f"| Genre: {movie['genre_similarity']:.3f} "
            f"| Rating: {movie['rating']:.1f} "
            f"| Final: {movie['similarity']:.3f}"
        )