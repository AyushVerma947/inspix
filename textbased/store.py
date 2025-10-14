import os
import faiss
import numpy as np
import whisper
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Directories
VIDEOS_DIR = "videos"
AUDIO_DIR = "audio"
TRANSCRIPTS_DIR = "transcripts"
FAISS_DB_DIR = "faiss_db"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Load models
print("🔄 Loading models...")
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Models loaded.")

# FAISS and metadata paths
FAISS_INDEX_PATH = os.path.join(FAISS_DB_DIR, "index_all.index")
METADATA_PATH = os.path.join(FAISS_DB_DIR, "metadata_all.pkl")


def load_faiss_and_metadata(dimension):
    """Load FAISS index and metadata if exist, else create new ones."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"📦 Loaded existing FAISS index with {len(metadata)} entries.")
    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        print("🆕 Created new FAISS index.")
    return index, metadata


def save_faiss_and_metadata(index, metadata):
    """Save FAISS index and metadata."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print("💾 FAISS and metadata saved.")


def extract_audio(video_path, audio_path):
    """Extract audio from video."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="mp3")
    clip.close()


def transcribe_audio(audio_path):
    """Transcribe audio using local Whisper model."""
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def get_embedding(text):
    """Generate local embeddings."""
    emb = embed_model.encode([text], convert_to_numpy=True)[0]
    return np.array(emb, dtype=np.float32)


def store_in_faiss(chunks, video_id, index, metadata):
    """Add new chunks to existing FAISS index."""
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
        metadata.append({
            "video_id": video_id,  # Keep track of which video this chunk belongs to
            "text": chunk
        })

    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)
    print(f"✅ Added {len(chunks)} chunks for video {video_id} to global index.")


def process_video(video_path, index, metadata):
    """Full pipeline for one video."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.txt")

    print(f"\n🎬 Processing {video_id}...")

    extract_audio(video_path, audio_path)
    print("🎧 Audio extracted.")

    text = transcribe_audio(audio_path)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("📝 Transcription complete.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)
    print(f"🔹 Created {len(chunks)} chunks with overlap.")

    store_in_faiss(chunks, video_id, index, metadata)


def main():
    """Process all videos and build single FAISS database."""
    # Get embedding dimension dynamically
    dim = len(get_embedding("test"))
    index, metadata = load_faiss_and_metadata(dim)

    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".mkv", ".mov")):
            video_path = os.path.join(VIDEOS_DIR, file)
            process_video(video_path, index, metadata)

    save_faiss_and_metadata(index, metadata)
    print("🎯 All videos processed and indexed successfully.")


if __name__ == "__main__":
    main()
