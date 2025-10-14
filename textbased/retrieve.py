import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_DB_DIR = "faiss_db"

# Load the same local embedding model used during indexing
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Convert text to local embedding vector."""
    return np.array(embed_model.encode([text], convert_to_numpy=True)[0], dtype=np.float32)

def query_video(video_id, query_text, top_k=3):
    """Retrieve top matching chunks for a given video and query."""
    index_path = os.path.join(FAISS_DB_DIR, f"{video_id}.index")
    meta_path = os.path.join(FAISS_DB_DIR, f"{video_id}_meta.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"❌ No FAISS index or metadata found for video: {video_id}")
        return []

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Generate query embedding
    query_vec = get_embedding(query_text).reshape(1, -1)

    # Search top_k similar chunks
    distances, indices = index.search(query_vec, top_k)

    print(f"\n🔍 Query: {query_text}")
    print(f"🎬 Searching inside video: {video_id}\n")

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk_info = metadata[idx]
            result = {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "text": chunk_info["text"],
                "video_id": chunk_info["video_id"]
            }
            results.append(result)
            print(f"Result {rank + 1}:")
            print(f"Score: {distances[0][rank]:.4f}")
            print(f"Text: {chunk_info['text'][:300]}...\n")

    return results

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    video_id = "EV Car Ownership Review in India - Things You Need to KNOW!".strip()
    query = "what does the reviewer say about charging time?".strip()
    query_video(video_id, query, top_k=3)
