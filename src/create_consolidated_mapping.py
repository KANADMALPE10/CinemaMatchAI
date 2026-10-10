from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

mapping_path = (
    ROOT / "data/processed/movie_catalog_mapping_approved.csv"
)

output_path = (
    ROOT / "data/processed/movie_catalog_mapping_consolidated.csv"
)

mapping = pd.read_csv(mapping_path)

# Keep only rows that have a TMDB match.
matched = mapping[mapping["tmdb_id"].notna()].copy()

# Group MovieLens entries that map to the same TMDB movie.
# Preserve every original MovieLens ID in the consolidated record.
consolidated = (
    matched.groupby("tmdb_id", as_index=False)
    .agg(
        tmdb_title=("tmdb_title", "first"),
        release_year=("release_year", "first"),
        movielens_ids=("movie_id", lambda x: ",".join(map(str, x))),
        movielens_titles=("title", lambda x: " | ".join(x)),
        original_entry_count=("movie_id", "count"),
    )
)

consolidated["tmdb_id"] = consolidated["tmdb_id"].astype(int)

consolidated.to_csv(output_path, index=False)

print("=" * 60)
print("CINEMAMATCH AI - CONSOLIDATED MOVIE MAPPING")
print("=" * 60)

print(f"Original MovieLens entries: {len(mapping)}")
print(f"Matched MovieLens entries: {len(matched)}")
print(f"Unique TMDB movies: {len(consolidated)}")
print(
    "Entries consolidated:",
    len(matched) - len(consolidated),
)

print("\nMovies with multiple MovieLens entries:")
duplicates = consolidated[
    consolidated["original_entry_count"] > 1
]

if duplicates.empty:
    print("None")
else:
    print(
        duplicates[
            [
                "tmdb_title",
                "tmdb_id",
                "movielens_ids",
                "original_entry_count",
            ]
        ].to_string(index=False)
    )

print(f"\nSaved to: {output_path}")