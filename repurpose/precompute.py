import os
import faiss
import whisper
import pickle
import numpy as np
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================

VIDEOS_DIR = "../textbased/videos"
AUDIO_DIR = "audio"
TRANSCRIPTS_DIR = "transcripts"
FAISS_DB_DIR = "faiss_db"

FAISS_INDEX_PATH = os.path.join(FAISS_DB_DIR, "index_all.index")
METADATA_PATH = os.path.join(FAISS_DB_DIR, "metadata_all.pkl")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# ===================== MODELS =====================

print("🔄 Loading models...")
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Models loaded")

# ===================== HELPERS =====================

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="mp3", logger=None)
    clip.close()


def get_embedding(text):
    emb = embed_model.encode([text], convert_to_numpy=True)[0]
    return np.array(emb, dtype=np.float32)


def load_faiss(dimension):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"📦 Loaded FAISS index with {len(metadata)} entries")
    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        print("🆕 Created new FAISS index")

    return index, metadata


def save_faiss(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print("💾 FAISS index + metadata saved")


# ===================== CORE PIPELINE =====================

def process_video(video_path, index, metadata):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")

    print(f"\n🎬 Processing video: {video_id}")

    # 1️⃣ Extract audio
    extract_audio(video_path, audio_path)
    print("🎧 Audio extracted")

    # 2️⃣ Whisper transcription WITH timestamps
    result = whisper_model.transcribe(audio_path)
    segments = result["segments"]

    print(f"📝 Found {len(segments)} whisper segments")

    embeddings = []

    # 3️⃣ Embed each segment and store metadata
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        emb = get_embedding(text)
        embeddings.append(emb)

        metadata.append({
            "video_id": video_id,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text
        })

    if embeddings:
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        print(f"✅ Added {len(embeddings)} segments to FAISS")

    # Optional: save raw transcript
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])


def main():
    # Get embedding dimension
    dim = len(get_embedding("test"))

    index, metadata = load_faiss(dim)

    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".mkv", ".mov")):

            video_id = os.path.splitext(file)[0]
            audio_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")

            if os.path.exists(audio_path):
                print(f"⏭️ Skipping {video_id}, audio already exists")
                continue
            
            process_video(os.path.join(VIDEOS_DIR, file), index, metadata)

    save_faiss(index, metadata)
    print("\n🎯 All videos indexed with timestamps successfully")


# ===================== ENTRY =====================

if __name__ == "__main__":
    main()
