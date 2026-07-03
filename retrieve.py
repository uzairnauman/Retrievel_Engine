import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    local_dir="models/bge"
)

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