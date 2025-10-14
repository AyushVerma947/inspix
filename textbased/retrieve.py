import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Directories
FAISS_DB_DIR = "faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DB_DIR, "index_all.index")
METADATA_PATH = os.path.join(FAISS_DB_DIR, "metadata_all.pkl")

# Load embedding model (same used during indexing)
print("🔄 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")

def get_embedding(text: str) -> np.ndarray:
    """Convert text to local embedding vector."""
    return np.array(embed_model.encode([text], convert_to_numpy=True)[0], dtype=np.float32)

def load_faiss_and_metadata():
    """Load FAISS index and metadata."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("❌ FAISS index or metadata file not found. Run the processing script first.")
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f"📦 Loaded FAISS index with {len(metadata)} total chunks.")
    return index, metadata

def query_video(video_id: str, query_text: str, top_k: int = 3):
    """Retrieve top matching chunks for a given video and query (from global index)."""
    print(f"\n🔍 Querying: '{query_text}' in video: {video_id}")

    # Load index + metadata
    index, metadata = load_faiss_and_metadata()

    # Filter metadata for given video_id
    video_chunks = [(i, m) for i, m in enumerate(metadata) if m["video_id"] == video_id]
    if not video_chunks:
        print(f"❌ No chunks found for video_id: {video_id}")
        return []

    # Extract embeddings of only this video's chunks from FAISS
    # Get their FAISS vector indices
    video_indices = np.array([i for i, _ in video_chunks])

    # Retrieve corresponding vectors from FAISS
    all_vectors = index.reconstruct_n(0, index.ntotal)
    video_vectors = all_vectors[video_indices]

    # Compute query embedding
    query_vec = get_embedding(query_text).reshape(1, -1)

    # Compute cosine similarity manually (since we’re using a subset)
    query_norm = query_vec / np.linalg.norm(query_vec)
    video_norms = video_vectors / np.linalg.norm(video_vectors, axis=1, keepdims=True)
    sims = np.dot(video_norms, query_norm.T).squeeze()

    # Sort top_k results
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    print(f"\n🎯 Top {top_k} results for video '{video_id}':\n")
    for rank, idx in enumerate(top_indices):
        chunk_info = video_chunks[idx][1]
        score = float(sims[idx])
        results.append({
            "rank": rank + 1,
            "score": score,
            "text": chunk_info["text"],
            "video_id": chunk_info["video_id"]
        })
        print(f"Result {rank + 1}:")
        print(f"Score: {score:.4f}")
        print(f"Text: {chunk_info['text'][:300]}...\n")

    return results

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    video_id = "137848"  # Replace with your actual video filename (ID)
    query = "what does the reviewer say about charging time?"
    query_video(video_id, query, top_k=3)
