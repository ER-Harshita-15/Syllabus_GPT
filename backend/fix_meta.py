from chromadb import PersistentClient

VECTOR_DB_DIR = "./vector-db"
client = PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_collection("study_kb")

# --------------------------------------------
# Improved subject detection using filename patterns
# --------------------------------------------

def detect_subject_from_filename(filename: str) -> str:
    name = filename.lower()

    # 🔵 AI
    if (
        "ai" in name or
        "artificial-intelligence" in name or
        "artificial intelligence" in name
    ):
        return "AI"

    # 🔵 ML
    if (
        "ml" in name or
        "machine learning" in name or
        "machine-learning" in name or
        "data science" in name or
        "data-science" in name or
        "introduction to machine learning" in name
    ):
        return "ML"

    # 🔵 IOT
    if "iot" in name or "internet of things" in name:
        return "IOT"

    # 🔵 TOC
    if (
        "toc" in name or
        "theoryofcomputation" in name or
        "theory of computation" in name
    ):
        return "TOC"

    # 🔵 STDS / STATS
    if (
        "stds" in name or
        "thinkstats" in name or
        "statistics" in name or
        "stats" in name
    ):
        return "STDS"

    return "UNKNOWN"


# --------------------------------------------
# 🚀 FIX METADATA
# --------------------------------------------

def fix_metadata():
    print("🔍 Fetching stored metadata...")

    items = collection.get(
        include=["metadatas", "documents"],
        limit=999999  # get all
    )

    all_meta = items["metadatas"]
    ids = items["ids"]

    updates = 0

    for idx, meta in enumerate(all_meta):
        old_sub = meta.get("subject", "UNKNOWN")
        filename = meta.get("source", "")

        corrected = detect_subject_from_filename(filename)

        if corrected != old_sub:
            print(f"Fixing → {filename}: {old_sub} → {corrected}")

            collection.update(
                ids=[ids[idx]],
                metadatas=[{
                    "subject": corrected,
                    "type": meta.get("type", "BOOK"),
                    "source": filename
                }]
            )

            updates += 1

    print(f"\n✅ DONE — Updated {updates} entries!")


if __name__ == "__main__":
    fix_metadata()
