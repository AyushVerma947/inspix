import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip, concatenate_videoclips

# ================= CONFIG =================

FAISS_INDEX_PATH = "faiss_db/index.faiss"
METADATA_PATH = "faiss_db/metadata.pkl"

SIMILARITY_THRESHOLD = 0.3
MERGE_GAP_SECONDS = 2
PADDING_SECONDS = 1

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD =================

index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= CORE =================

def generate_clip(video_id: str, query: str):
    output_path = os.path.join(
        OUTPUT_DIR, f"{video_id}_query_clip.mp4"
    )

    query_emb = model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    sentence_embeddings = index.reconstruct_n(0, index.ntotal)

    similarities = cosine_similarity(
        [query_emb],
        sentence_embeddings
    )[0]

    matches = []

    for i, score in enumerate(similarities):
        if score >= SIMILARITY_THRESHOLD:
            meta = metadata[i]
            if meta["video_id"] != video_id:
                continue

            matches.append({
                "video_path": meta["video_path"],
                "start": meta["start"],
                "end": meta["end"]
            })

    if not matches:
        print("❌ No matches found")
        return

    matches.sort(key=lambda x: x["start"])

    merged = [matches[0]]
    for m in matches[1:]:
        prev = merged[-1]
        if m["start"] - prev["end"] <= MERGE_GAP_SECONDS:
            prev["end"] = max(prev["end"], m["end"])
        else:
            merged.append(m)

    clips = []
    for seg in merged:
        video = VideoFileClip(seg["video_path"])
        s = max(0, seg["start"] - PADDING_SECONDS)
        e = min(video.duration, seg["end"] + PADDING_SECONDS)
        clips.append(video.subclip(s, e))

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac"
    )

    print(f"🎯 Saved → {output_path}")

# ================= MAIN =================

if __name__ == "__main__":
    generate_clip(
        video_id="142536",
        query="tell me about battery "
    )
