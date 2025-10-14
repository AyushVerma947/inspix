import os
import faiss
import pickle
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ============================================================== #
# CONFIGURATION
# ============================================================== #
FAISS_DB_DIR = "faiss_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 1000  # for hierarchical summarization
TOP_K = 3          # top chunks to retrieve for QA

print("🔄 Loading models...")

# Summarizer (BART)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    device=-1 if DEVICE == "cpu" else 0
)

# Embedding model for vector search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Stronger LLM for Q&A (Flan-T5 Large)
qa_model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(DEVICE)

print(f"✅ All models loaded successfully on {DEVICE}!\n")

# ============================================================== #
# HELPER FUNCTIONS
# ============================================================== #

def get_embedding(text):
    return np.array(embedder.encode([text], convert_to_numpy=True)[0], dtype=np.float32)

def load_faiss_data():
    index_path = os.path.join(FAISS_DB_DIR, "index_all.index")
    meta_path = os.path.join(FAISS_DB_DIR, "metadata_all.pkl")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("❌ FAISS index or metadata file not found!")
    index_all = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata_all = pickle.load(f)
    print(f"📦 Loaded FAISS index with {len(metadata_all)} total chunks.")
    return index_all, metadata_all

def get_video_chunks(video_id, index_all, metadata_all):
    filtered_indices = [i for i, m in enumerate(metadata_all) if str(m["video_id"]) == str(video_id)]
    if not filtered_indices:
        return np.array([]), []
    embeddings = np.array([index_all.reconstruct(i) for i in filtered_indices], dtype=np.float32)
    metadata = [metadata_all[i] for i in filtered_indices]
    return embeddings, metadata

def hierarchical_summarize(texts, chunk_size=CHUNK_SIZE):
    if not texts:
        return "No content available to summarize."
    full_text = " ".join(texts)
    subchunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    summaries = []
    print(f"🔹 Split transcript into {len(subchunks)} chunks.")
    for i, chunk in enumerate(subchunks):
        print(f"  ▶ Summarizing chunk {i+1}/{len(subchunks)}...")
        try:
            summary = summarizer(chunk, max_length=200, min_length=40, truncation=True)[0]['summary_text']
        except Exception as e:
            summary = f"[Error summarizing chunk {i+1}] {str(e)}"
        summaries.append(summary)
    # Combine summaries and summarize again
    combined_text = " ".join(summaries)
    try:
        final_summary = summarizer(combined_text, max_length=300, min_length=80, truncation=True)[0]['summary_text']
    except Exception as e:
        final_summary = combined_text
    return final_summary

def retrieve_context(query, embeddings, metadata, top_k=TOP_K):
    if embeddings.size == 0:
        return []
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, min(top_k, len(metadata)))
    return [metadata[idx]["text"] for idx in indices[0]]

def answer_query(question, context_chunks):
    """
    Use chunked context and strong LLM to answer questions reliably.
    """
    best_answer = "Not found in the video."
    for chunk in context_chunks:
        prompt = (
            f"Context:\n{chunk}\n\n"
            f"Question: {question}\n"
            "Answer only using the context above. "
            "If the answer is not present in the context, respond 'Not found in the video.'"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        outputs = qa_model.generate(**inputs, max_length=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if answer.lower() not in ["not found in the video", ""]:
            best_answer = answer
            break
    return best_answer

# ============================================================== #
# MAIN FUNCTION
# ============================================================== #

def summarize_and_chat(video_id):
    print(f"🎥 Selected video: {video_id}\n")
    try:
        index_all, metadata_all = load_faiss_data()
    except FileNotFoundError as e:
        print(str(e))
        return

    embeddings, metadata = get_video_chunks(video_id, index_all, metadata_all)
    if not metadata:
        print(f"❌ No transcript metadata found for video: {video_id}")
        return

    print("🧠 Generating hierarchical summary...")
    texts = [m["text"] for m in metadata]
    summary = hierarchical_summarize(texts)
    print("\n✅ Summary Generated:\n")
    print(summary)
    print("\n" + "="*80)

    # Q&A loop
    while True:
        user_query = input("\nAsk a question about this video (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break
        elif "summary" in user_query.lower():
            print("\n📄 Summary:\n", summary)
            continue

        context_chunks = retrieve_context(user_query, embeddings, metadata)
        if not context_chunks:
            print("❌ No relevant context found for this question.")
            continue

        print("\n🤖 Answer:")
        answer = answer_query(user_query, context_chunks)
        print(answer)

# ============================================================== #
# ENTRY POINT
# ============================================================== #

if __name__ == "__main__":
    video_id = "137848"  # Replace with your video ID
    summarize_and_chat(video_id)
