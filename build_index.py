import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Quran JSON
# -------------------------
with open("data/quran.json", "r", encoding="utf-8") as f:
    quran = json.load(f)

documents = []

for surah in quran:
    for verse in surah["verses"]:
        documents.append({
            "surah": surah["id"],
            "surah_name": surah["transliteration"],
            "surah_arabic": surah["name"],
            "surah_translation": surah["translation"],
            "type": surah["type"],
            "ayah": verse["id"],
            "arabic": verse["text"],
            "english": verse["translation"],
        })

print(f"Loaded {len(documents)} verses.")

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------
# Create embeddings
# -------------------------
texts = [doc["english"] for doc in documents]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings shape:", embeddings.shape)

# -------------------------
# Build FAISS index
# -------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"Indexed {index.ntotal} verses.")

# -------------------------
# Save index
# -------------------------
faiss.write_index(index, "data/index.faiss")

with open("data/metadata.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index saved.")
print("Metadata saved.")