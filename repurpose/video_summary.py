import faiss
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# ================= CONFIG =================

FAISS_INDEX_PATH = "faiss_db/index.faiss"
METADATA_PATH = "faiss_db/metadata.pkl"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 6
TOP_SENTENCES_PER_CLUSTER = 2
MERGE_GAP_SECONDS = 2
PADDING_SECONDS = 1


# ================= LOAD DATABASE =================

def load_vector_db():
    print("🔄 Loading FAISS + metadata...")
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded {len(metadata)} total sentences")
    return index, metadata


# ================= FILTER VIDEO DATA =================

def get_video_sentences(index, metadata, video_id):
    indices = [i for i, m in enumerate(metadata) if m["video_id"] == video_id]

    if not indices:
        raise ValueError(f"❌ Video {video_id} not found")

    embeddings = np.vstack([index.reconstruct(i) for i in indices])
    meta = [metadata[i] for i in indices]
    video_path = meta[0]["video_path"]

    print(f"🎬 Found {len(meta)} sentences for video {video_id}")
    return embeddings, meta, video_path


# ================= CLUSTER + SELECT IMPORTANT SENTENCES =================

def select_representative_segments(embeddings, meta):
    print("🔍 Clustering sentences...")

    k = min(N_CLUSTERS, len(embeddings))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    selected = []

    for cid in range(k):
        cluster_idxs = np.where(labels == cid)[0]
        if len(cluster_idxs) == 0:
            continue

        cluster_embs = embeddings[cluster_idxs]
        centroid = kmeans.cluster_centers_[cid]

        sims = cosine_similarity([centroid], cluster_embs)[0]
        top_local = np.argsort(sims)[-TOP_SENTENCES_PER_CLUSTER:]

        for idx in top_local:
            m = meta[cluster_idxs[idx]]
            selected.append({"start": m["start"], "end": m["end"]})

    print(f"✅ Selected {len(selected)} key segments")
    return selected


# ================= MERGE NEARBY SEGMENTS =================

def merge_segments(segments):
    segments.sort(key=lambda x: x["start"])

    merged = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]

        if seg["start"] - last["end"] <= MERGE_GAP_SECONDS:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg)

    print(f"🔗 Merged into {len(merged)} segments")
    return merged


# ================= BUILD FINAL VIDEO =================

def build_summary_video(video_path, segments, output_path):
    print("✂️ Extracting clips...")
    video = VideoFileClip(video_path)

    clips = []
    for seg in segments:
        s = max(0, seg["start"] - PADDING_SECONDS)
        e = min(video.duration, seg["end"] + PADDING_SECONDS)
        clips.append(video.subclip(s, e))

    final = concatenate_videoclips(clips, method="compose")

    print("💾 Writing summary video...")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"🎯 Saved at: {output_path}")


# ================= MAIN PIPELINE =================

def generate_summary(video_id):
    index, metadata = load_vector_db()

    embeddings, meta, video_path = get_video_sentences(
        index, metadata, video_id
    )

    selected_segments = select_representative_segments(
        embeddings, meta
    )

    merged_segments = merge_segments(selected_segments)

    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_summary.mp4")

    build_summary_video(video_path, merged_segments, output_path)


# ================= ENTRY POINT =================

if __name__ == "__main__":
    video_id = "137848"   # 👈 change this
    generate_summary(video_id)
