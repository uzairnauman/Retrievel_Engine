import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import snapshot_download
import urllib.request

snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    local_dir="models/bge"
)

# PASTE YOUR DIRECT GOOGLE DRIVE LINKS HERE
# ---------------------------------------------------------
FAISS_URL = "https://docs.google.com/uc?export=download&id=1RQ88p02NWUNOA2n5K7ANpI0_67L5-QGN&confirm=t"
METADATA_URL = "https://docs.google.com/uc?export=download&id=1mXpCWq3iowWvfEC9eu6ZTPxnGNJYRSm1&confirm=t""

# 3. Auto-download index file if missing on the server
if not os.path.exists("data/index.faiss"):
    print("Downloading FAISS index from Google Drive...")
    urllib.request.urlretrieve(FAISS_URL, "data/index.faiss")

# 4. Auto-download metadata file if missing on the server
if not os.path.exists("data/metadata.pkl"):
    print("Downloading metadata from Google Drive...")
    urllib.request.urlretrieve(METADATA_URL, "data/metadata.pkl")

# -------------------------
# Load FAISS index
# -------------------------
index = faiss.read_index("data/index.faiss")

# -------------------------
# Load metadata
# -------------------------
with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer(
    "models/bge"
)


def search(query, k=5):
    # Create embedding for the query
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search FAISS
    scores, indices = index.search(embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        doc = metadata[idx]
        results.append({
            "score": float(score),
            "surah": doc["surah"],
            "surah_name": doc["surah_name"],
            "ayah": doc["ayah"],
            "arabic": doc["arabic"],
            "english": doc["english"]
        })

    return results


# -------------------------
# Test
# -------------------------
results = search("What does quran say about patience?")

for r in results:
    print("=" * 60)
    print(f"Similarity : {r['score']:.4f}")
    print(f"Surah {r['surah']} ({r['surah_name']}) Ayah {r['ayah']}")
    print("Arabic:")
    print(r["arabic"])
    print("English:")
    print(r["english"])
