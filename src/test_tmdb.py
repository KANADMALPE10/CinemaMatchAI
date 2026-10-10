import os

import requests
from dotenv import load_dotenv

# Load environment variables from the project root
load_dotenv()

# Retrieve the TMDB API Read Access Token
access_token = os.getenv("TMDB_ACCESS_TOKEN")

if not access_token:
    raise ValueError(
        "TMDB_ACCESS_TOKEN not found. "
        "Check your .env file."
    )

# TMDB popular movies endpoint
url = "https://api.themoviedb.org/3/movie/popular"

# Authentication headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "accept": "application/json",
}

# Request parameters
params = {
    "language": "en-US",
    "page": 1,
}

print("Connecting to TMDB...")

# Send API request
response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30,
)

# Check for HTTP errors
response.raise_for_status()

# Parse response
data = response.json()

# Display results
movies = data.get("results", [])

print("\nTMDB connection successful!")
print(f"Movies retrieved: {len(movies)}")
print(f"Total pages available: {data.get('total_pages')}")
print("\nFirst 5 movies:\n")

for index, movie in enumerate(movies[:5], start=1):
    print(f"{index}. {movie['title']}")
    print(f"   Release date: {movie.get('release_date') or 'Unknown'}")
    print(f"   Rating: {movie.get('vote_average', 'N/A')}")
    print()