import os
import time
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

TOKEN = os.getenv("TMDB_ACCESS_TOKEN")

if not TOKEN:
    raise ValueError("TMDB_ACCESS_TOKEN missing from .env")


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tmdb_catalog.csv"
)

URL = "https://api.themoviedb.org/3/discover/movie"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "accept": "application/json",
}


# --------------------------------------------------
# 2. Configure retries
# --------------------------------------------------

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=2,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)
session.headers.update(HEADERS)


# --------------------------------------------------
# 3. Collect popular movies across decades
# --------------------------------------------------

START_YEAR = 1980
END_YEAR = 2026

# Discover supports pagination, but TMDB limits the
# maximum page number for a query.
MAX_PAGES_PER_YEAR = 20

movies_by_id = {}

print("=" * 60)
print("CINEMATCH AI - EXPANDED MOVIE CATALOG")
print("=" * 60)

for year in range(START_YEAR, END_YEAR + 1):

    print(f"\nCollecting movies from {year}...")

    for page in range(1, MAX_PAGES_PER_YEAR + 1):

        params = {
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "primary_release_date.gte": f"{year}-01-01",
            "primary_release_date.lte": f"{year}-12-31",
            "page": page,
        }

        try:
            response = session.get(
                URL,
                params=params,
                timeout=(15, 45),
            )

            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as error:
            print(
                f"Request failed for {year}, page {page}: {error}"
            )
            break

        results = data.get("results", [])

        if not results:
            break

        for movie in results:

            movie_id = movie.get("id")

            if not movie_id:
                continue

            movies_by_id[movie_id] = {
                "tmdb_id": movie_id,
                "title": movie.get("title", ""),
                "original_title": movie.get("original_title", ""),
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", ""),
                "genre_ids": ",".join(
                    str(g) for g in movie.get("genre_ids", [])
                ),
                "vote_average": movie.get("vote_average", 0),
                "vote_count": movie.get("vote_count", 0),
                "popularity": movie.get("popularity", 0),
                "poster_path": movie.get("poster_path", ""),
                "original_language": movie.get("original_language", ""),
                "adult": movie.get("adult", False),
            }

        # Avoid making requests too rapidly.
        time.sleep(0.1)

        # Stop if we've reached the final available page.
        if page >= data.get("total_pages", 1):
            break

    print(
        f"Unique movies collected so far: {len(movies_by_id)}"
    )


# --------------------------------------------------
# 4. Save the catalog
# --------------------------------------------------

df = pd.DataFrame(movies_by_id.values())

if df.empty:
    raise RuntimeError(
        "No movies were collected. Check the network and TMDB API."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 60)
print("COLLECTION COMPLETED")
print("=" * 60)

print(f"Total unique movies: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nMovies by release year:")

years = pd.to_datetime(
    df["release_date"],
    errors="coerce",
).dt.year

print(
    years.value_counts()
    .sort_index()
    .tail(15)
    .to_string()
)