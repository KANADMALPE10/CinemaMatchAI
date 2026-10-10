import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOVIES_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.item"
)

CANDIDATES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_matching_review.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_matching_approval.csv"
)

# Load MovieLens movie metadata.
movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "movielens_title"],
)

# Load the proposed fuzzy matches.
candidates = pd.read_csv(CANDIDATES_PATH)

# Keep only candidates that require review.
candidates = candidates[
    candidates["match_status"] == "fuzzy_candidate_review_required"
].copy()

# Add the original MovieLens titles.
review = candidates.merge(
    movies,
    on="movie_id",
    how="left",
)

review = review[
    [
        "movie_id",
        "movielens_title",
        "candidate_tmdb_title",
        "candidate_tmdb_year",
        "candidate_tmdb_id",
        "match_confidence",
    ]
].copy()

# The previous script saved the candidate TMDB ID in
# candidate_tmdb_id only if that column exists.
# Otherwise, preserve the report and add the column.
if "candidate_tmdb_id" not in review.columns:
    review["candidate_tmdb_id"] = pd.NA

# Add an explicit manual decision column.
review["approved"] = ""

# Keep the report easy to review.
review = review[
    [
        "movie_id",
        "movielens_title",
        "candidate_tmdb_title",
        "candidate_tmdb_year",
        "candidate_tmdb_id",
        "match_confidence",
        "approved",
    ]
]

review = review.sort_values(
    "match_confidence",
    ascending=False,
)

review.to_csv(OUTPUT_PATH, index=False)

print("=" * 65)
print("CINEMAMATCH AI - MATCH APPROVAL REPORT")
print("=" * 65)

print(f"\nCandidates requiring review: {len(review)}")

print("\nREPORT PREVIEW")
print("-" * 65)

print(review.to_string(index=False))

print(f"\nApproval report saved to:\n{OUTPUT_PATH}")