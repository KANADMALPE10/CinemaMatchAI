from pathlib import Path
import re

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.improved_content_recommender import ImprovedContentRecommender


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "catalog_cleaned.csv"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="CinemaMatch AI API",
    description=(
        "Movie discovery and content-based recommendation API"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MOVIES AND RECOMMENDER
# ============================================================

if not DATA_PATH.exists():
    raise RuntimeError(
        f"Movie catalog not found: {DATA_PATH}"
    )

movies = pd.read_csv(DATA_PATH)

movies["tmdb_id"] = pd.to_numeric(
    movies["tmdb_id"],
    errors="coerce",
)

movies = movies.dropna(
    subset=["tmdb_id", "title"]
).copy()

movies["tmdb_id"] = movies["tmdb_id"].astype(int)

movies = (
    movies.drop_duplicates("tmdb_id")
    .reset_index(drop=True)
)

print("Loading CinemaMatch AI recommendation model...")

recommender = ImprovedContentRecommender()

# Fast lookup for movie details.
movie_lookup = {
    int(row["tmdb_id"]): row
    for _, row in movies.iterrows()
}

print(f"Loaded {len(movies):,} movies.")
print("CinemaMatch AI backend is ready.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_number(value, default=0.0):
    """Convert a value to a finite float."""

    try:
        number = float(value)

        if pd.isna(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def movie_to_dict(row):
    """Convert a catalog record into JSON-friendly data."""

    release_date = row.get("release_date")

    if pd.isna(release_date):
        year = None
    else:
        year = str(release_date)[:4]

    poster_path = row.get("poster_path")

    if pd.isna(poster_path):
        poster_path = None

    vote_count = safe_number(
        row.get("vote_count", 0)
    )

    return {
        "tmdb_id": int(row["tmdb_id"]),
        "title": str(row.get("title", "")),
        "year": year,
        "rating": round(
            safe_number(row.get("vote_average", 0)),
            1,
        ),
        "genres": str(row.get("genres", "") or ""),
        "overview": str(row.get("overview", "") or ""),
        "poster_path": poster_path,
        "vote_count": int(vote_count),
    }


def get_movie_or_404(tmdb_id):
    """Find a movie or raise a 404 response."""

    movie = movie_lookup.get(tmdb_id)

    if movie is None:
        raise HTTPException(
            status_code=404,
            detail="Movie not found in the catalog.",
        )

    return movie


def split_genres(value):
    """
    Split genre strings.

    Supports the catalog's space-separated genres as well as
    pipe- or comma-separated strings, if encountered.
    """

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Most entries in the current catalog use spaces.
    # Match known multi-word genre names first.
    known_genres = [
        "Science Fiction",
        "TV Movie",
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Thriller",
        "War",
        "Western",
    ]

    if "|" in text:
        parts = text.split("|")

    elif "," in text:
        parts = text.split(",")

    else:
        # Parse space-separated genres while preserving
        # multi-word names such as Science Fiction.
        remaining = text
        parts = []

        for genre in sorted(
            known_genres,
            key=len,
            reverse=True,
        ):

            pattern = rf"(?<!\w){re.escape(genre)}(?!\w)"

            if re.search(
                pattern,
                remaining,
                flags=re.IGNORECASE,
            ):

                parts.append(genre)

                remaining = re.sub(
                    pattern,
                    " ",
                    remaining,
                    flags=re.IGNORECASE,
                )

        # Retain any unrecognized genre labels.
        parts.extend(
            item.strip()
            for item in remaining.split()
            if item.strip()
        )

    # Remove duplicates while preserving order.
    result = []
    seen = set()

    for item in parts:

        item = item.strip()

        if not item:
            continue

        key = item.casefold()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# ============================================================
# GENERAL ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to CinemaMatch AI",
        "status": "running",
        "movies_available": len(movies),
        "recommendation_model": "improved_content_recommender",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "movies_loaded": len(movies),
        "content_model_loaded": True,
        "recommendation_model": "improved_content_recommender",
    }


# ============================================================
# SEARCH MOVIES
# ============================================================

@app.get("/api/movies")
def search_movies(
    q: str = Query(default="", max_length=100),
    limit: int = Query(default=24, ge=1, le=100),
):
    """
    Search the real movie catalog by title.

    With no query, return popular movies first.
    """

    results = movies

    if q.strip():

        results = results[
            results["title"].astype(str).str.contains(
                q.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    results = results.sort_values(
        ["vote_count", "vote_average", "popularity"],
        ascending=[False, False, False],
        na_position="last",
    ).head(limit)

    return {
        "count": len(results),
        "results": [
            movie_to_dict(row)
            for _, row in results.iterrows()
        ],
    }


# ============================================================
# MOVIE DETAILS
# ============================================================

@app.get("/api/movies/{tmdb_id}")
def movie_details(tmdb_id: int):
    """Return information about one movie."""

    row = get_movie_or_404(tmdb_id)

    return movie_to_dict(row)


# ============================================================
# IMPROVED RECOMMENDATIONS
# ============================================================

@app.get("/api/movies/{tmdb_id}/similar")
def similar_movies(
    tmdb_id: int,
    limit: int = Query(default=12, ge=1, le=50),
):
    """
    Recommend similar movies using the improved content model.

    Ranking combines:
    - Story similarity
    - Genre similarity
    - Existing content similarity
    - Configured franchise relationships
    """

    get_movie_or_404(tmdb_id)

    try:

        recommendations = recommender.recommend(
            tmdb_id=tmdb_id,
            limit=limit,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"Recommendation error for movie {tmdb_id}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendations.",
        ) from exc

    results = []

    for recommendation in recommendations:

        # Start with the actual catalog record so the response
        # includes the same fields as the movie-search endpoint.
        movie_id = int(recommendation["tmdb_id"])

        catalog_row = movie_lookup.get(movie_id)

        if catalog_row is not None:
            result = movie_to_dict(catalog_row)
        else:
            result = dict(recommendation)

        # Add recommendation-specific information.
        result["similarity"] = recommendation["similarity"]

        result["overview_similarity"] = (
            recommendation["overview_similarity"]
        )

        result["genre_similarity"] = (
            recommendation["genre_similarity"]
        )

        result["franchise_boost"] = (
            recommendation["franchise_boost"]
        )

        results.append(result)

    return {
        "source_movie_id": tmdb_id,
        "count": len(results),
        "results": results,
    }


# ============================================================
# GENRES
# ============================================================

@app.get("/api/genres")
def get_genres():
    """Return individual genre names found in the catalog."""

    genres = set()

    for value in movies["genres"].dropna():

        for genre in split_genres(value):

            if genre:
                genres.add(genre)

    return {
        "genres": sorted(
            genres,
            key=str.casefold,
        ),
    }