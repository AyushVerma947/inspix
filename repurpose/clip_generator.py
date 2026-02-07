import os
import faiss
import pickle
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================

FAISS_INDEX_PATH = "faiss_db/index_all.index"
METADATA_PATH = "faiss_db/metadata_all.pkl"
VIDEOS_DIR = "../textbased/videos"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===================== LOAD FAISS =====================

def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ===================== SEARCH =====================

def embed_query(text):
    return embed_model.encode([text], convert_to_numpy=True).astype("float32")


def search_segments(video_id, query, top_k=20):
    index, metadata = load_faiss()
    q_emb = embed_query(query)

    D, I = index.search(q_emb, top_k)

    segments = []
    for idx in I[0]:
        m = metadata[idx]
        if m["video_id"] == video_id:
            segments.append(m)

    return segments


# ===================== MERGE TIMESTAMPS =====================

def merge_segments(segments, max_gap=1.0):
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"] + max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return merged


# ===================== VIDEO GENERATION =====================

def generate_video(video_id, segments):
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    video = VideoFileClip(video_path)

    clips = []
    for seg in segments:
        start = max(0, seg["start"] - 0.5)
        end = min(video.duration, seg["end"] + 0.5)
        clips.append(video.subclip(start, end))

    final = concatenate_videoclips(clips, method="compose")

    output_path = os.path.join(
        OUTPUT_DIR, f"{video_id}_query_result.mp4"
    )

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac"
    )

    video.close()
    return output_path


# ===================== END-TO-END =====================

def query_to_video(video_id, query):
    print(f"\n🔍 Query: {query}")

    segments = search_segments(video_id, query)

    if not segments:
        print("❌ No relevant segments found")
        return

    print(f"✅ Found {len(segments)} raw segments")

    merged = merge_segments(segments)
    print(f"✂️ Merged into {len(merged)} clips")

    output = generate_video(video_id, merged)
    print(f"🎉 Final video saved at: {output}")


# ===================== USAGE =====================

if __name__ == "__main__":
    query_to_video(
        video_id="142536",
        query="battery, camera"
    )

