import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOVIELENS_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.item"
)

TMDB_PATH = (
    PROJECT_ROOT / "data" / "processed" / "catalog_cleaned.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_catalog_mapping.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_catalog_mapping_improved.csv"
)


# ============================================================
# 2. TITLE NORMALIZATION
# ============================================================

def normalize_title(title):
    if pd.isna(title):
        return ""

    title = str(title).casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title)

    return " ".join(title.split())


def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))

    if match:
        return int(match.group(1))

    return None


def clean_movielens_title(title):
    title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title),
    ).strip()

    # Move trailing articles to the beginning.
    match = re.match(
        r"^(.*),\s*(The|A|An)$",
        title,
        flags=re.IGNORECASE,
    )

    if match:
        title = f"{match.group(2)} {match.group(1)}"

    # Replace parenthesized alternative titles with a space
    # instead of removing them without a separator.
    title = re.sub(r"\([^)]*\)", " ", title)

    # Normalize whitespace.
    title = " ".join(title.split())

    return title

# ============================================================
# 3. LOAD DATA
# ============================================================

movielens = pd.read_csv(
    MOVIELENS_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

tmdb = pd.read_csv(TMDB_PATH)

tmdb["release_year"] = pd.to_numeric(
    tmdb["release_year"],
    errors="coerce",
)

# Preserve the existing mapping and its exact matches.
mapping = pd.read_csv(MAPPING_PATH)

print("=" * 70)
print("CINEMAMATCH AI - IMPROVED CATALOG MATCHING")
print("=" * 70)

# ============================================================
# 4. PREPARE MATCHING REPRESENTATIONS
# ============================================================

movielens["release_year"] = movielens["title"].apply(extract_year)

movielens["normalized_title"] = (
    movielens["title"]
    .apply(clean_movielens_title)
    .apply(normalize_title)
)

tmdb["normalized_title"] = tmdb["title"].apply(normalize_title)

# Use only unique TMDB title-year combinations for candidate matching.
tmdb_candidate_counts = (
    tmdb.groupby(
        ["normalized_title", "release_year"],
        dropna=False,
    )["tmdb_id"]
    .nunique()
    .reset_index(name="candidate_count")
)

tmdb = tmdb.merge(
    tmdb_candidate_counts,
    on=["normalized_title", "release_year"],
    how="left",
)

# Create a lookup grouped by release year.
tmdb_by_year = {
    int(year): group.reset_index(drop=True)
    for year, group in tmdb.dropna(
        subset=["release_year"]
    ).groupby("release_year")
}


# ============================================================
# 5. KEEP EXISTING EXACT MATCHES
# ============================================================

mapping["match_confidence"] = None

exact_mask = (
    mapping["match_status"] == "matched_unique_title_year"
)

mapping.loc[exact_mask, "match_confidence"] = 1.0

# Create a lookup for each MovieLens movie.
mapping_by_id = mapping.set_index("movie_id")


# ============================================================
# 6. FUZZY MATCH UNMATCHED MOVIES
# ============================================================

new_matches = []

for row in movielens.itertuples(index=False):

    movie_id = row.movie_id

    existing = mapping_by_id.loc[movie_id]

    if existing["match_status"] == "matched_unique_title_year":
        continue

    year = row.release_year
    normalized_title = row.normalized_title

    if pd.isna(year) or int(year) not in tmdb_by_year:
        new_matches.append(
            {
                "movie_id": movie_id,
                "match_status": "unmatched",
                "match_confidence": 0.0,
                "candidate_tmdb_id": None,
                "candidate_tmdb_title": None,
                "candidate_tmdb_year": None,
            }
        )
        continue

    candidates = tmdb_by_year[int(year)].copy()

    scored = []

    for candidate in candidates.itertuples(index=False):

        similarity = SequenceMatcher(
            None,
            normalized_title,
            candidate.normalized_title,
        ).ratio()

        scored.append(
            {
                "tmdb_id": candidate.tmdb_id,
                "tmdb_title": candidate.title,
                "tmdb_year": candidate.release_year,
                "similarity": similarity,
                "candidate_count": candidate.candidate_count,
            }
        )

    scored.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    if not scored:
        new_matches.append(
            {
                "movie_id": movie_id,
                "match_status": "unmatched",
                "match_confidence": 0.0,
                "candidate_tmdb_id": None,
                "candidate_tmdb_title": None,
                "candidate_tmdb_year": None,
            }
        )
        continue

    best = scored[0]

    # Only propose strong matches where the title-year combination
    # is unique. Lower-confidence candidates remain unmatched.
    if (
        best["similarity"] >= 0.90
        and best["candidate_count"] == 1
    ):
        status = "fuzzy_candidate_review_required"
    else:
        status = "unmatched"

    new_matches.append(
        {
            "movie_id": movie_id,
            "match_status": status,
            "match_confidence": round(best["similarity"], 4),
            "candidate_tmdb_id": best["tmdb_id"],
            "candidate_tmdb_title": best["tmdb_title"],
            "candidate_tmdb_year": best["tmdb_year"],
        }
    )


# ============================================================
# 7. SAVE REVIEWABLE CANDIDATES
# ============================================================

new_matches_df = pd.DataFrame(new_matches)

review_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movie_matching_review.csv"
)

new_matches_df.to_csv(review_path, index=False)

print(f"\nExisting exact matches: {exact_mask.sum():,}")

print(
    "Strong fuzzy candidates requiring review: "
    f"{(new_matches_df['match_status'] == 'fuzzy_candidate_review_required').sum():,}"
)

print(
    "Still unmatched: "
    f"{(new_matches_df['match_status'] == 'unmatched').sum():,}"
)

print("\nSAMPLE FUZZY CANDIDATES")
print("-" * 70)

review_candidates = new_matches_df[
    new_matches_df["match_status"] == "fuzzy_candidate_review_required"
].head(20)

if review_candidates.empty:
    print("No strong fuzzy candidates were found.")
else:
    print(
        review_candidates[
            [
                "movie_id",
                "match_confidence",
                "candidate_tmdb_title",
                "candidate_tmdb_year",
            ]
        ].to_string(index=False)
    )

print(f"\nReview file saved to:\n{review_path}")
print("\nImproved matching analysis completed.")