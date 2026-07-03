import json

# Load Quran JSON
with open("data/quran.json", "r", encoding="utf-8") as f:
    quran = json.load(f)

documents = []

# Flatten the nested structure
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

# Show a few examples
for doc in documents[:5]:
    print(doc)