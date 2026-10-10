import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOVIELENS_PATH = (
    PROJECT_ROOT / "data" / "ratings" / "ml-100k" / "u.item"
)

TMDB_PATH = (
    PROJECT_ROOT / "data" / "processed" / "catalog_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "movie_catalog_mapping.csv"


# ============================================================
# 2. NORMALIZE TITLES
# ============================================================

def normalize_title(title):
    """Normalize a movie title for cautious title matching."""

    if pd.isna(title):
        return ""

    title = str(title).casefold()

    # Normalize punctuation and whitespace.
    title = re.sub(r"[^a-z0-9]+", " ", title)

    return " ".join(title.split())


def extract_year(title):
    """Extract a four-digit year in parentheses from MovieLens titles."""

    match = re.search(r"\((\d{4})\)\s*$", str(title))

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 3. LOAD MOVIELENS MOVIES
# ============================================================

movielens = pd.read_csv(
    MOVIELENS_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)

movielens["release_year"] = movielens["title"].apply(
    extract_year
)

# MovieLens commonly places leading articles at the end:
# "Shawshank Redemption, The (1994)".
# Move the trailing article to the beginning for matching.
def fix_movielens_title(title):
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()

    match = re.match(
        r"^(.*),\s*(The|A|An)$",
        title,
        flags=re.IGNORECASE,
    )

    if match:
        title = f"{match.group(2)} {match.group(1)}"

    return title


movielens["normalized_title"] = (
    movielens["title"]
    .apply(fix_movielens_title)
    .apply(normalize_title)
)


# ============================================================
# 4. LOAD TMDB CATALOG
# ============================================================

tmdb = pd.read_csv(TMDB_PATH)

required_columns = {
    "tmdb_id",
    "title",
    "release_year",
}

missing_columns = required_columns - set(tmdb.columns)

if missing_columns:
    raise ValueError(
        f"TMDB catalog is missing columns: {missing_columns}"
    )

tmdb["release_year"] = pd.to_numeric(
    tmdb["release_year"],
    errors="coerce",
).astype("Int64")

tmdb["normalized_title"] = tmdb["title"].apply(
    normalize_title
)


# ============================================================
# 5. MATCH BY NORMALIZED TITLE AND YEAR
# ============================================================

# Remove duplicate TMDB title-year combinations before merging.
# If several TMDB records share a title and year, we will flag
# the ambiguity rather than silently choosing one.

tmdb_key_counts = (
    tmdb.groupby(
        ["normalized_title", "release_year"],
        dropna=False,
    )["tmdb_id"]
    .nunique()
    .rename("tmdb_match_count")
    .reset_index()
)

tmdb_unique = tmdb.merge(
    tmdb_key_counts,
    on=["normalized_title", "release_year"],
    how="left",
)

tmdb_unique_keys = tmdb_unique[
    tmdb_unique["tmdb_match_count"] == 1
].copy()

tmdb_ambiguous_keys = tmdb_unique[
    tmdb_unique["tmdb_match_count"] > 1
][
    ["normalized_title", "release_year"]
].drop_duplicates()

# Match only unique title-year combinations.
matched = movielens.merge(
    tmdb_unique_keys[
        [
            "normalized_title",
            "release_year",
            "tmdb_id",
            "title",
            "tmdb_match_count",
        ]
    ].rename(
        columns={"title": "tmdb_title"}
    ),
    on=["normalized_title", "release_year"],
    how="left",
    suffixes=("_movielens", "_tmdb"),
)

matched["match_status"] = "unmatched"

matched.loc[
    matched["tmdb_id"].notna(),
    "match_status",
] = "matched_unique_title_year"

ambiguous_mask = matched["tmdb_id"].isna() & matched[
    ["normalized_title", "release_year"]
].apply(tuple, axis=1).isin(
    set(
        map(
            tuple,
            tmdb_ambiguous_keys[
                ["normalized_title", "release_year"]
            ].to_numpy(),
        )
    )
)

matched.loc[
    ambiguous_mask,
    "match_status",
] = "ambiguous_title_year"


# ============================================================
# 6. SAVE AND REPORT
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matched.to_csv(
    OUTPUT_PATH,
    index=False,
)

total = len(matched)
matched_count = (matched["match_status"] == "matched_unique_title_year").sum()
ambiguous_count = (matched["match_status"] == "ambiguous_title_year").sum()
unmatched_count = (matched["match_status"] == "unmatched").sum()

print("=" * 70)
print("CINEMAMATCH AI - MOVIE CATALOG MATCHING")
print("=" * 70)

print(f"\nMovieLens movie records: {total:,}")
print(f"Unique title-year matches: {matched_count:,}")
print(f"Ambiguous title-year records: {ambiguous_count:,}")
print(f"Unmatched records: {unmatched_count:,}")

print(
    f"Unique-match coverage: "
    f"{matched_count / total:.2%}"
)

print("\nSAMPLE MATCHES")
print("-" * 70)

sample_matches = matched[
    matched["match_status"] == "matched_unique_title_year"
].head(15)

print(
    sample_matches[
        [
            "movie_id",
            "title",
            "release_year",
            "tmdb_id",
            "tmdb_title",
        ]
    ].to_string(index=False)
)

print("\nSAMPLE UNMATCHED MOVIES")
print("-" * 70)

sample_unmatched = matched[
    matched["match_status"] == "unmatched"
].head(15)

print(
    sample_unmatched[
        ["movie_id", "title", "release_year"]
    ].to_string(index=False)
)

print(f"\nMapping saved to:\n{OUTPUT_PATH}")
print("\nCatalog matching completed.")