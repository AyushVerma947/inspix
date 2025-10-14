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

# Load models (all local)
print("🔄 Loading models...")
whisper_model = whisper.load_model("base")  # Local Whisper
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Local embedding model
print("✅ Models loaded.")

def extract_audio(video_path, audio_path):
    """Extract audio from video."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='mp3')
    clip.close()

def transcribe_audio(audio_path):
    """Transcribe audio using local Whisper model."""
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def get_embedding(text):
    """Generate local embeddings."""
    emb = embed_model.encode([text], convert_to_numpy=True)[0]
    return np.array(emb, dtype=np.float32)

def store_in_faiss(chunks, video_id):
    """Store chunks in FAISS index with metadata."""
    dimension = len(get_embedding("test"))
    index = faiss.IndexFlatL2(dimension)
    embeddings = []
    metadata = []

    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
        metadata.append({
            "video_id": video_id,
            "text": chunk
        })

    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)

    faiss.write_index(index, f"{FAISS_DB_DIR}/{video_id}.index")
    with open(f"{FAISS_DB_DIR}/{video_id}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Stored {len(chunks)} chunks for video {video_id}")

def process_video(video_path):
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

    store_in_faiss(chunks, video_id)

def main():
    """Process all videos in the folder."""
    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".mkv", ".mov")):
            video_path = os.path.join(VIDEOS_DIR, file)
            process_video(video_path)

if __name__ == "__main__":
    main()
