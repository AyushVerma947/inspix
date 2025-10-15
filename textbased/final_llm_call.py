import os
import faiss
import pickle
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
# Sentiment & Emotion Analysis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import json

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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

# Sentiment Analysis (fine-tuned RoBERTa)
# cardiffnlp/twitter-roberta-base-sentiment-latest
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(DEVICE)

flan_model_name = "google/flan-t5-large"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name).to(DEVICE)


# Aspect-Based Sentiment (optional: same model but applied on extracted aspects)
aspect_model_name = sentiment_model_name  # reuse same model for ABSA
aspect_tokenizer = sentiment_tokenizer
aspect_model = sentiment_model
aspect_model = pipeline("text2text-generation", model=flan_model, tokenizer=tokenizer)


# =======================================================
# Define Pydantic models for structured parsing
# =======================================================

class AspectExtractionModel(BaseModel):
    aspects: list[str] = Field(..., description="List of main aspects/features mentioned in the text")

class AspectSentimentModel(BaseModel):
    aspect: str = Field(..., description="The product/service aspect")
    sentiment: str = Field(..., description="Sentiment label: positive, negative, or neutral")

# Create parsers
aspect_parser = PydanticOutputParser(pydantic_object=AspectExtractionModel)
sentiment_parser = PydanticOutputParser(pydantic_object=AspectSentimentModel)


# ============================================================== #
# HELPER FUNCTIONS
# ============================================================== #

def analyze_sentiment(text):
    """
    Returns sentiment label and confidence score for a given text.
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs, dim=1).item()
    label = sentiment_model.config.id2label[label_id]
    score = probs[0][label_id].item()
    return {"label": label, "score": round(score, 3)}

def aspect_based_sentiment_analysis(text, aspects):
    """
    Performs simple Aspect-Based Sentiment Analysis by checking sentiment of each aspect mention.
    """
    results = {}
    for aspect in aspects:
        if aspect.lower() in text.lower():
            inputs = aspect_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = aspect_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(probs, dim=1).item()
            label = aspect_model.config.id2label[label_id]
            results[aspect] = label
    return results if results else {"info": "No specified aspects found."}


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


# ----------- 1️⃣ Aspect Extraction with Parser -----------
def extract_aspects_simple(video_id, index_all, metadata_all, chunk_size=CHUNK_SIZE):
    """
    Extract aspects from video chunks using FLAN-T5.
    """
    # video_chunks = [m['text'] for m in metadata_all if m['video_id'] == video_id]
    # if not video_chunks:
    #     print("No transcript available for this video.")
    #     return ""
    
    # full_text = " ".join(video_chunks)
    # subchunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    # all_aspects_text = ""
    # print(f"🔹 Split transcript of video {video_id} into {len(subchunks)} chunks.")
    
    # for i, chunk in enumerate(subchunks):
    #     print(f"  ▶ Extracting aspects from chunk {i+1}/{len(subchunks)}...")
    #     try:
    #         prompt = (
    #             "Extract all product/service aspects, features, or attributes mentioned in the text. "
    #             "Return them as a comma-separated list. Only include explicit aspects, no opinions.\n\n"
    #             f"Text: {chunk}"
    #         )
    #         result = aspect_model(prompt, max_length=200)[0]['generated_text']
    #         all_aspects_text += result + " "
    #     except Exception as e:
    #         print(f"    ⚠ Skipping chunk {i+1} due to error: {e}")
    
    # final_text = all_aspects_text.strip()
    # print(f"✅ Final collected aspects for video {video_id}:")
    # # Save to a temp file
    temp_file = "temp_final_text.txt"
    # with open(temp_file, "w", encoding="utf-8") as f:
    #     f.write(final_text)

    # Read back from the file
    with open(temp_file, "r", encoding="utf-8") as f:
        final_text_from_file = f.read()

    print("\n✅ Final text read from file:")
    print(final_text_from_file)
    # print(final_text)
    # After collecting all chunk-level outputs in final_text# Hierarchical aspect-sentiment extraction using pipeline
    import re

    if final_text_from_file:
        aspects_text = final_text_from_file.strip().lower()
        
        print("\n🔹 Running aspect-sentiment extraction...")
        
        # Define aspect patterns and sentiment indicators
        aspect_patterns = {
            'battery': ['battery', 'kwh', 'kw pack'],
            'performance': ['bhp', 'torque', 'nm', 'power', 'acceleration'],
            'range': ['range', 'km', 'kilometers', 'charge'],
            'handling': ['handling', 'steering', 'suspension', 'dampers', 'body roll'],
            'comfort': ['comfort', 'ride quality', 'seats', 'ventilated'],
            'interior': ['interior', 'dashboard', 'upholstery', 'cabin', 'screens'],
            'exterior': ['exterior', 'design', 'fascia', 'led', 'headlamp', 'logo'],
            'technology': ['screen', 'software', 'snapdragon', 'responsive', 'soc'],
            'audio': ['speakers', 'audio', 'harmon', 'dolby', 'watts'],
            'storage': ['storage', 'boot', 'space', 'liters', 'capacity'],
            'price': ['price', 'cost', 'value', 'lakh', 'cheaper']
        }
        
        positive_words = [
            'impressive', 'excellent', 'best', 'good', 'love', 'stunning', 'amazing',
            'great', 'superb', 'fantastic', 'wonderful', 'perfect', 'beautiful'
        ]
        
        negative_words = [
            'con', 'issue', 'miss', 'lack', 'poor', 'bad', 'dull', 'problem',
            'disappointing', 'worse', 'cheap', 'misses out', 'major miss'
        ]
        
        results = []
        
        for aspect, keywords in aspect_patterns.items():
            # Check if aspect is mentioned
            aspect_mentioned = any(kw in aspects_text for kw in keywords)
            
            if aspect_mentioned:
                # Find context around aspect mentions
                sentiment_score = 0
                
                for keyword in keywords:
                    if keyword in aspects_text:
                        # Get 100 characters around the keyword
                        idx = aspects_text.find(keyword)
                        context_start = max(0, idx - 100)
                        context_end = min(len(aspects_text), idx + 100)
                        context = aspects_text[context_start:context_end]
                        
                        # Count positive/negative words in context
                        pos_count = sum(1 for pw in positive_words if pw in context)
                        neg_count = sum(1 for nw in negative_words if nw in context)
                        
                        sentiment_score += pos_count - neg_count
                
                # Determine overall sentiment
                if sentiment_score > 0:
                    sentiment = 'positive'
                elif sentiment_score < 0:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                results.append((aspect, sentiment))
        
        # Print in perfect format - line by line
        print("\n" + "="*60)
        print("✅ Final Hierarchical Aspects with Sentiment")
        print("="*60)
        
        for aspect, sentiment in results:
            # Capitalize aspect name and format nicely
            print(f"{aspect.capitalize():.<20} {sentiment}")
        
        print("="*60)
        print(f"📊 Total Aspects Extracted: {len(results)}")
        print("="*60)
        
        # Also save as comma-separated (if needed)
        output = ", ".join([f"{aspect}: {sentiment}" for aspect, sentiment in results])
        print(f"\n📋 Comma-separated format:\n{output}")


def extract_aspects_from_text(chunk, model=flan_model, tokenizer=flan_tokenizer, device=DEVICE):
    """
    Extract key product/service aspects dynamically using Flan-T5.
    Returns a list of unique aspects.
    """
    prompt_str = (
        "Identify the main aspects, features, or attributes discussed "
        "in the following review or transcript text.\n\n"
        f"Text:\n{chunk}\n\n"
        "Return ONLY a JSON array of unique aspects. Example: [\"battery\", \"design\", \"price\"]"
    )

    inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Attempt to extract JSON array
    aspects = []
    try:
        # Try parsing directly
        aspects = json.loads(response)
        if not isinstance(aspects, list):
            aspects = []
    except json.JSONDecodeError:
        # fallback: regex to find array-like string
        import re
        match = re.search(r'\[.*\]', response)
        if match:
            try:
                aspects = json.loads(match.group())
            except:
                aspects = []

    # Clean and deduplicate
    aspects = [a.lower().strip() for a in aspects if isinstance(a, str)]
    return list(set(aspects))

# ----------- 2️⃣ ABSA with Parser -----------
def run_absa_with_flan_dynamic_aspects(video_id, index_all, metadata_all, chunk_size=CHUNK_SIZE):
    """
    Performs hierarchical Aspect-Based Sentiment Analysis (ABSA)
    for a specific video using Flan-T5 with structured JSON prompts.
    Automatically extracts aspects from the transcript.
    """
    embeddings, metadata = get_video_chunks(video_id, index_all, metadata_all)
    if not metadata:
        print(f"❌ No transcript metadata found for video: {video_id}")
        return

    print(f"🎬 Running ABSA for video {video_id} with {len(metadata)} transcript chunks...")
    texts = [m["text"] for m in metadata]
    full_text = " ".join(texts)
    subchunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"🔹 Split transcript into {len(subchunks)} chunks for ABSA")

    all_aspect_results = []
    all_extracted_aspects = set()

    for i, chunk in enumerate(subchunks):
        print(f"  ▶ Processing chunk {i+1}/{len(subchunks)} for aspect extraction...")

        # Extract aspects dynamically using parser
        aspects = extract_aspects_from_text(chunk)
        if not aspects:
            print(f"    ⚠️ No aspects detected in chunk {i+1}")
            continue

        all_extracted_aspects.update(aspects)

        # ABSA prompt
        prompt_json = {
            "task": "Aspect-Based Sentiment Analysis",
            "instruction": (
                "Analyze the given text and determine sentiment (positive, negative, neutral) "
                "for each aspect listed. Return strictly in JSON format with aspect → sentiment mapping."
            ),
            "aspects": aspects,
            "text": chunk
        }

        prompt_str = (
            f"You are a sentiment analysis model.\n"
            f"Here is the input JSON:\n{json.dumps(prompt_json, indent=2)}\n\n"
            "Return strictly as JSON in the format: {\"aspect\": \"sentiment\", ...}"
        )

        inputs = flan_tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = flan_model.generate(**inputs, max_length=400)
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Parse sentiment JSON safely using Pydantic parser
        try:
            chunk_results = json.loads(response)
        except json.JSONDecodeError:
            chunk_results = {}

        all_aspect_results.append(chunk_results)

    # Aggregate results
    print("🧩 Aggregating aspect sentiments across all chunks...")
    aspect_summary = {a: [] for a in all_extracted_aspects}
    for r in all_aspect_results:
        if isinstance(r, dict):
            for aspect in all_extracted_aspects:
                if aspect in r:
                    aspect_summary[aspect].append(r[aspect].lower())

    final_absa = {}
    for aspect, sentiments in aspect_summary.items():
        if not sentiments:
            continue
        dominant = max(["positive", "negative", "neutral"], key=lambda s: sentiments.count(s))
        confidence = round(sentiments.count(dominant) / len(sentiments), 2)
        final_absa[aspect] = {"sentiment": dominant, "confidence": confidence}

    # Display
    print("\n💬 Final Aspect-Based Sentiment Summary:")
    for aspect, info in final_absa.items():
        print(f" - {aspect.capitalize()}: {info['sentiment']} (Confidence: {info['confidence']})")

    return final_absa

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
    # summary = hierarchical_summarize(texts)
    # print("\n✅ Summary Generated:\n")
    # print(summary)
    # print("\n" + "="*80)

    # Sentiment analysis on full video transcript
    print("🧩 Running sentiment analysis on full transcript...")
    full_transcript = " ".join([m["text"] for m in metadata])
    sentiment_result = analyze_sentiment(full_transcript)
    print(f"\n🧠 Overall Sentiment: {sentiment_result['label']} (Confidence: {sentiment_result['score']})")

    # Aspect-Based Sentiment
    common_aspects = ["product", "service", "performance", "battery", "design", "price", "experience"]
    # absa_results = aspect_based_sentiment_analysis(full_transcript, common_aspects)
    # absa_results = run_absa_with_flan_dynamic_aspects(video_id, index_all, metadata_all)
    extract_aspects_simple(video_id, index_all, metadata_all)
    return
    print("\n💬 Aspect-Based Sentiment:")
    for aspect, sentiment in absa_results.items():
        print(f" - {aspect.capitalize()}: {sentiment}")


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
