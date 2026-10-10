import os
import json
import time
import requests

from pathlib import Path
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")

if not TMDB_ACCESS_TOKEN:
    raise ValueError(
        "TMDB_ACCESS_TOKEN was not found in the .env file."
    )


# --------------------------------------------------
# 2. Configure HTTP retries
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

session.headers.update({
    "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
    "accept": "application/json",
})


# --------------------------------------------------
# 3. Request genres from TMDB
# --------------------------------------------------

URL = "https://api.themoviedb.org/3/genre/movie/list"

print("Connecting to TMDB...")
print("Requesting movie genres...")

try:
    response = session.get(
        URL,
        params={"language": "en-US"},
        timeout=(15, 45),
    )

    response.raise_for_status()

    data = response.json()

except requests.exceptions.SSLError as error:
    print("\nSSL connection error.")
    print("Check whether a VPN, proxy, or network filter is interfering.")
    print(f"Details: {error}")
    raise SystemExit(1)

except requests.exceptions.ConnectionError as error:
    print("\nCould not establish a stable connection to TMDB.")
    print("Check your internet connection and try again.")
    print(f"Details: {error}")
    raise SystemExit(1)

except requests.exceptions.Timeout:
    print("\nThe request timed out. Please try again.")
    raise SystemExit(1)

except requests.exceptions.HTTPError as error:
    print("\nTMDB returned an HTTP error.")
    print(f"Status code: {response.status_code}")
    print(f"Details: {error}")
    raise SystemExit(1)


# --------------------------------------------------
# 4. Create genre mapping
# --------------------------------------------------

genre_mapping = {
    str(genre["id"]): genre["name"]
    for genre in data["genres"]
}

print(f"\nGenres retrieved successfully: {len(genre_mapping)}")

for genre_id, genre_name in sorted(
    genre_mapping.items(),
    key=lambda item: int(item[0]),
):
    print(f"{genre_id}: {genre_name}")


# --------------------------------------------------
# 5. Save genre mapping
# --------------------------------------------------

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "genre_mapping.json"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        genre_mapping,
        file,
        indent=4,
        ensure_ascii=False,
    )

print("\nGenre mapping saved successfully!")
print(f"Location: {OUTPUT_PATH}")