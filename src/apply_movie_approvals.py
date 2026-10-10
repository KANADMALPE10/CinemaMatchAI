from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

approval_path = ROOT / "data/processed/movie_matching_approval.csv"
original_path = ROOT / "data/processed/movie_catalog_mapping.csv"
output_path = ROOT / "data/processed/movie_catalog_mapping_approved.csv"

approvals = pd.read_csv(approval_path)
original = pd.read_csv(original_path)

# Explicitly approved candidates with matching titles and release years.
approved_ids = [
    16, 121, 1652, 1632, 1462,
    1385, 1240, 1199, 1196, 1195,
    1193, 1006, 958, 1682, 730,
    529, 462, 306, 198, 166
]

# Preserve the original mapping and add approved candidates separately.
result = original.copy()

for _, row in approvals.iterrows():
    if int(row["movie_id"]) not in approved_ids:
        continue

    movie_id = int(row["movie_id"])
    tmdb_id = int(row["candidate_tmdb_id"])

    mask = result["movie_id"] == movie_id

    if mask.any():
        result.loc[mask, "tmdb_id"] = tmdb_id
        result.loc[mask, "match_status"] = "approved_fuzzy_match"

result.to_csv(output_path, index=False)

print(f"Original mapping rows: {len(original)}")
print(f"Approved candidates applied: {len(approved_ids)}")
print(f"Updated mapping saved to: {output_path}")
print("\nMatch statuses:")
print(result["match_status"].value_counts(dropna=False))
