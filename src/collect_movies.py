import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise ValueError(
        "TMDB_ACCESS_TOKEN not found. "
        "Check the .env file in your project root."
    )


BASE_URL = "https://api.themoviedb.org/3/discover/movie"

START_DATE = "2026-01-01"
END_DATE = "2026-10-09"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "tmdb_movies_2026.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. API CONFIGURATION
# --------------------------------------------------

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "accept": "application/json",
}

params = {
    "language": "en-US",
    "region": "US",
    "sort_by": "primary_release_date.desc",
    "primary_release_date.gte": START_DATE,
    "primary_release_date.lte": END_DATE,
    "include_adult": "false",
    "page": 1,
}

session = requests.Session()
session.headers.update(headers)


# --------------------------------------------------
# 3. FETCH FIRST PAGE
# --------------------------------------------------

print("Connecting to TMDB...")
print(f"Fetching movies released from {START_DATE} to {END_DATE}")

response = session.get(
    BASE_URL,
    params=params,
    timeout=30,
)

response.raise_for_status()

first_page = response.json()

total_pages = min(first_page.get("total_pages", 0), 500)

print(f"Pages available: {first_page.get('total_pages', 0)}")
print(f"Pages to collect: {total_pages}")


# --------------------------------------------------
# 4. FETCH ALL SELECTED PAGES
# --------------------------------------------------

movies = []
seen_ids = set()

for page in range(1, total_pages + 1):

    params["page"] = page

    try:
        response = session.get(
            BASE_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        for movie in data.get("results", []):

            movie_id = movie.get("id")

            if movie_id is None or movie_id in seen_ids:
                continue

            seen_ids.add(movie_id)

            movies.append({
                "tmdb_id": movie_id,
                "title": movie.get("title"),
                "original_title": movie.get("original_title"),
                "release_date": movie.get("release_date"),
                "overview": movie.get("overview"),
                "genre_ids": ",".join(
                    str(genre_id)
                    for genre_id in movie.get("genre_ids", [])
                ),
                "vote_average": movie.get("vote_average"),
                "vote_count": movie.get("vote_count"),
                "popularity": movie.get("popularity"),
                "poster_path": movie.get("poster_path"),
                "original_language": movie.get("original_language"),
                "adult": movie.get("adult"),
            })

        print(
            f"Page {page}/{total_pages} | "
            f"Unique movies collected: {len(movies)}"
        )

        time.sleep(0.1)

    except requests.RequestException as error:
        print(f"Error collecting page {page}: {error}")
        print("Stopping collection to avoid losing track of failed pages.")
        break


# --------------------------------------------------
# 5. SAVE DATASET
# --------------------------------------------------

df = pd.DataFrame(movies)

if not df.empty:

    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce",
    )

    df = df.drop_duplicates(subset=["tmdb_id"])

    df = df.sort_values(
        by="release_date",
        ascending=False,
        na_position="last",
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCollection completed!")
    print(f"Movies saved: {len(df)}")
    print(f"Dataset location: {OUTPUT_FILE}")
    print(f"Columns: {list(df.columns)}")

else:
    print("No movie records were collected.")

session.close()