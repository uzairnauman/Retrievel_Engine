import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
import gdown
from huggingface_hub import snapshot_download

# 1. Download the embedding model from Hugging Face
snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    local_dir="models/bge"
)

# 2. Ensure the local data directory exists on the server!
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# GOOGLE DRIVE FILE IDS
# ---------------------------------------------------------
FAISS_ID = "1mXpCWq3iowWvfEC9eu6ZTPxnGNJYRSm1"
METADATA_ID = "1RQ88p02NWUNOA2n5K7ANpI0_67L5-QGN"

# FORCE CLEAN: Kill any broken files sitting in the cache completely
for filename in ["index.faiss", "metadata.pkl"]:
    filepath = os.path.join("data", filename)
    if os.path.exists(filepath):
        print(f"Purging cached copy of {filename} to ensure fresh download...")
        os.remove(filepath)

# 3. Download files cleanly via gdown
print("Downloading FAISS index from Google Drive via gdown...")
gdown.download(id=FAISS_ID, output="data/index.faiss", quiet=False)

print("Downloading metadata from Google Drive via gdown...")
gdown.download(id=METADATA_ID, output="data/metadata.pkl", quiet=False)

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
