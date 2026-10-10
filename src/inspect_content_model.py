from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
matrix_path = MODEL_DIR / "tfidf_matrix.joblib"
metadata_path = MODEL_DIR / "movie_metadata.joblib"

print("=" * 65)
print("CINEMAMATCH AI - CONTENT MODEL INSPECTION")
print("=" * 65)

# Verify that the saved model files exist.
for path in [
    vectorizer_path,
    matrix_path,
    metadata_path,
]:
    print(f"{path.name}: {'FOUND' if path.exists() else 'MISSING'}")

# Load the saved artifacts.
vectorizer = joblib.load(vectorizer_path)
tfidf_matrix = joblib.load(matrix_path)
metadata = joblib.load(metadata_path)

print("\nMODEL DETAILS")
print("-" * 40)

print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Metadata shape: {metadata.shape}")

print("\nMETADATA COLUMNS")
print("-" * 40)
print(metadata.columns.tolist())

print("\nFIRST FIVE MOVIES")
print("-" * 40)

columns_to_show = [
    column
    for column in [
        "tmdb_id",
        "title",
        "release_year",
        "genres",
    ]
    if column in metadata.columns
]

print(metadata[columns_to_show].head().to_string(index=False))

print("\nContent model inspection completed.")