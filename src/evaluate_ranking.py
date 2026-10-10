from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

PREDICTIONS_PATH = DATA_DIR / "learned_hybrid_predictions.csv"
OUTPUT_PATH = DATA_DIR / "ranking_evaluation.csv"

K = 10

# A rating of 4 or 5 counts as relevant.
RELEVANT_RATING_THRESHOLD = 4.0

# Evaluate only users with at least one relevant test movie.
MIN_RELEVANT_ITEMS = 1


# ============================================================
# 2. LOAD TEST PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTIONS_PATH)

required_columns = {
    "user_id",
    "tmdb_id",
    "rating",
    "collaborative_prediction",
    "hybrid_prediction",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df.duplicated(["user_id", "tmdb_id"]).any():
    raise ValueError(
        "Duplicate user/movie pairs found. "
        "Resolve duplicates before evaluating ranking."
    )

print("=" * 65)
print("CINEMAMATCH AI - TOP-10 RANKING EVALUATION")
print("=" * 65)

print(f"\nTest interactions: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Relevant rating threshold: {RELEVANT_RATING_THRESHOLD}")
print(f"Recommendation cutoff: {K}")


# ============================================================
# 3. DEFINE RANKING METRICS
# ============================================================

def precision_at_k(relevance, k):
    """
    Fraction of the top-k recommended items that are relevant.
    """

    top_k = relevance[:k]

    return float(np.sum(top_k) / k)


def recall_at_k(relevance, total_relevant, k):
    """
    Fraction of all relevant test items retrieved in the top-k.
    """

    if total_relevant == 0:
        return np.nan

    return float(np.sum(relevance[:k]) / total_relevant)


def ndcg_at_k(relevance, k):
    """
    Normalized Discounted Cumulative Gain at k.

    Relevant items ranked higher receive more credit.
    """

    relevance = np.asarray(relevance[:k], dtype=float)

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevance) + 2))

    dcg = np.sum(relevance / discounts)

    ideal_relevance = np.sort(relevance)[::-1]

    ideal_dcg = np.sum(ideal_relevance / discounts)

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)


# ============================================================
# 4. EVALUATE EACH USER
# ============================================================

results = []

for user_id, user_df in df.groupby("user_id"):

    # Relevant movies in this user's test interactions.
    relevant_items = set(
        user_df.loc[
            user_df["rating"] >= RELEVANT_RATING_THRESHOLD,
            "tmdb_id"
        ]
    )

    # Skip users without any relevant test items.
    if len(relevant_items) < MIN_RELEVANT_ITEMS:
        continue

    # Relevance labels for ranking metrics.
    user_df = user_df.copy()

    user_df["relevant"] = (
        user_df["rating"] >= RELEVANT_RATING_THRESHOLD
    ).astype(int)

    # Rank using collaborative-filtering predictions.
    cf_ranked = user_df.sort_values(
        "collaborative_prediction",
        ascending=False
    )

    # Rank using learned hybrid predictions.
    hybrid_ranked = user_df.sort_values(
        "hybrid_prediction",
        ascending=False
    )

    cf_relevance = cf_ranked["relevant"].to_numpy()

    hybrid_relevance = hybrid_ranked["relevant"].to_numpy()

    total_relevant = len(relevant_items)

    results.append({
        "user_id": user_id,

        "relevant_test_items": total_relevant,

        "cf_precision_at_10": precision_at_k(
            cf_relevance, K
        ),

        "cf_recall_at_10": recall_at_k(
            cf_relevance, total_relevant, K
        ),

        "cf_ndcg_at_10": ndcg_at_k(
            cf_relevance, K
        ),

        "hybrid_precision_at_10": precision_at_k(
            hybrid_relevance, K
        ),

        "hybrid_recall_at_10": recall_at_k(
            hybrid_relevance, total_relevant, K
        ),

        "hybrid_ndcg_at_10": ndcg_at_k(
            hybrid_relevance, K
        ),
    })


# ============================================================
# 5. SUMMARIZE RESULTS
# ============================================================

user_results = pd.DataFrame(results)

if user_results.empty:
    raise ValueError(
        "No users had relevant test items. "
        "Check the relevance threshold and test data."
    )

metric_pairs = [
    (
        "Precision@10",
        "cf_precision_at_10",
        "hybrid_precision_at_10",
    ),
    (
        "Recall@10",
        "cf_recall_at_10",
        "hybrid_recall_at_10",
    ),
    (
        "NDCG@10",
        "cf_ndcg_at_10",
        "hybrid_ndcg_at_10",
    ),
]

summary = []

for metric_name, cf_column, hybrid_column in metric_pairs:

    cf_score = user_results[cf_column].mean()
    hybrid_score = user_results[hybrid_column].mean()

    summary.append({
        "Metric": metric_name,
        "Collaborative Filtering": cf_score,
        "Learned Hybrid": hybrid_score,
        "Difference": hybrid_score - cf_score,
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 80)
print("RANKING RESULTS")
print("=" * 80)

print(f"\nUsers evaluated: {len(user_results):,}")

print(
    summary_df.to_string(
        index=False,
        formatters={
            "Collaborative Filtering": "{:.4f}".format,
            "Learned Hybrid": "{:.4f}".format,
            "Difference": "{:+.4f}".format,
        },
    )
)

# Save both the summary and user-level results.
user_results.to_csv(OUTPUT_PATH, index=False)

print(f"\nPer-user results saved to: {OUTPUT_PATH}")

print(
    "\nNote: These metrics evaluate ranking among each user's "
    "held-out test interactions, not the entire movie catalog."
)