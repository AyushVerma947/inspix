import os
import faiss
import pickle
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Directories
FAISS_DB_DIR = "faiss_db"

# Load embedding model (same used for storage)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load summarization model
print("🔄 Loading summarizer and LLM models (this may take a minute)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load small local LLM for Q&A
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("✅ Models loaded successfully!\n")


# ---------- Utility Functions ----------

def get_embedding(text):
    """Return vector embedding for given text."""
    return embed_model.encode([text])[0]


def load_faiss(video_id):
    """Load FAISS index and metadata for a specific video."""
    index_path = f"{FAISS_DB_DIR}/{video_id}.index"
    meta_path = f"{FAISS_DB_DIR}/{video_id}_meta.pkl"

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        print(f"❌ No FAISS index or metadata found for video: {video_id}")
        return None, None

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def query_video(video_id, question, top_k=3):
    """Retrieve most relevant transcript chunks from FAISS."""
    index, metadata = load_faiss(video_id)
    if index is None:
        return ""

    query_emb = np.array([get_embedding(question)]).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    results = [((1 / (1 + distances[0][i])), metadata[indices[0][i]]) for i in range(top_k)]
    results.sort(key=lambda x: x[0], reverse=True)

    context = "\n\n".join([meta["text"] for _, meta in results])
    return context


# ---------- Hierarchical Summarization ----------

def chunk_text(text, max_tokens=1500, overlap=200):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks


def hierarchical_summarize(full_text):
    """Perform hierarchical summarization over long transcripts."""
    chunks = chunk_text(full_text)
    print(f"🔹 Splitting transcript into {len(chunks)} chunks...")

    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Summarizing chunk {i}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=200, min_length=60, do_sample=False)[0]['summary_text']
        partial_summaries.append(summary)

    print("🧩 Combining partial summaries...")
    combined_text = " ".join(partial_summaries)

    print("🧠 Generating final hierarchical summary...")
    final_summary = summarizer(combined_text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
    return final_summary


def summarize_video(video_id):
    """Generate summary of entire video transcript."""
    meta_path = f"{FAISS_DB_DIR}/{video_id}_meta.pkl"
    if not os.path.exists(meta_path):
        print(f"❌ No transcript metadata found for video: {video_id}")
        return

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    full_text = " ".join([m["text"] for m in metadata])

    print("🧠 Generating hierarchical summary for video...\n")
    summary = hierarchical_summarize(full_text)

    print("\n🎬 Final Video Summary:\n")
    print(summary)
    return summary


# ---------- Question Answering ----------

def generate_answer(context, question):
    """Generate answer from local small LLM using retrieved context."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly and briefly."

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def answer_question(video_id, question):
    """Retrieve context and generate answer."""
    context = query_video(video_id, question)
    if not context:
        return "No context found."

    print("\n💬 Generating answer using local LLM...\n")
    return generate_answer(context, question)


# ---------- Main ----------

if __name__ == "__main__":
    video_id = "EV Car Ownership Review in India - Things You Need to KNOW!"

    print(f"🎥 Selected video: {video_id}\n")

    # Step 1: Generate hierarchical summary
    summarize_video(video_id)

    # Step 2: Ask user questions
    while True:
        q = input("\nAsk a question about this video (or 'exit' to quit): ").strip()
        if q.lower() == "exit":
            break

        answer = answer_question(video_id, q)
        print(f"\n🤖 Answer:\n{answer}\n")
