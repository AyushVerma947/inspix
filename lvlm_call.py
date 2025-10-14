# import os
# import shutil
# import torch
# import numpy as np
# from PIL import Image

# # =============================
# # VECTOR DB LIBRARIES
# # =============================
# from sentence_transformers import SentenceTransformer
# import open_clip
# import faiss

# # =============================
# # LLM LIBRARY
# # =============================
# import ollama  # pip install ollama

# # =============================
# # CONFIG
# # =============================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOP_K_TEXT = 3
# TOP_K_IMAGE = 3
# RESULTS_FOLDER = "results_images"

# # LVLM model on Ollama
# MODEL_NAME = "hf.co/NexaAI/OmniVLM-968M:Q8_0"


# # =============================
# # LOAD VECTOR DB MODELS
# # =============================
# # Text embeddings
# text_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
# TEXT_INDEX_FILE = "text_index.faiss"
# TEXT_METADATA_FILE = "text_metadata.npy"
# text_index = faiss.read_index(TEXT_INDEX_FILE)
# text_metadata = np.load(TEXT_METADATA_FILE, allow_pickle=True)

# # Image embeddings (CLIP)
# clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# clip_model = clip_model.to(DEVICE).eval()
# tokenizer = open_clip.get_tokenizer("ViT-B-32")
# IMAGE_INDEX_FILE = "image_index.faiss"
# IMAGE_METADATA_FILE = "image_metadata.npy"
# image_index = faiss.read_index(IMAGE_INDEX_FILE)
# image_metadata = np.load(IMAGE_METADATA_FILE, allow_pickle=True)

# # =============================
# # RETRIEVE CONTEXT
# # =============================
# def retrieve_context(query, top_k_text=TOP_K_TEXT, top_k_image=TOP_K_IMAGE):
#     # --- Text ---
#     q_emb_text = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
#     D_txt, I_txt = text_index.search(q_emb_text, top_k_text)
#     text_chunks = [text_metadata[idx]["content"] for idx in I_txt[0]]

#     # --- Images ---
#     tokens = tokenizer(query).to(DEVICE)
#     with torch.no_grad():
#         q_emb_img = clip_model.encode_text(tokens)
#         q_emb_img /= q_emb_img.norm(dim=-1, keepdim=True)
#     D_img, I_img = image_index.search(q_emb_img.cpu().numpy(), top_k_image)

#     os.makedirs(RESULTS_FOLDER, exist_ok=True)
#     # Clear previous results
#     for f in os.listdir(RESULTS_FOLDER):
#         os.remove(os.path.join(RESULTS_FOLDER, f))

#     retrieved_images = []
#     for rank, idx in enumerate(I_img[0]):
#         meta = image_metadata[idx]
#         if meta["type"] == "image":
#             src_path = os.path.join("key_frames", meta["file"])
#             if os.path.exists(src_path):
#                 dst_path = os.path.join(RESULTS_FOLDER, f"rank{rank+1}_{meta['file']}")
#                 shutil.copy(src_path, dst_path)
#                 retrieved_images.append(dst_path)

#     return text_chunks, retrieved_images

# # =============================
# # ASK LVLM
# # =============================
# def ask_omnivlm(query):
#     text_chunks, images = retrieve_context(query)

#     # --- Show retrieved context ---
#     print("\n=== Retrieved Text Chunks ===")
#     for i, chunk in enumerate(text_chunks, 1):
#         print(f"{i}. {chunk}\n")

#     print("=== Retrieved Images ===")
#     for img in images:
#         print(img)

#     # --- Prepare prompt ---
#     context_text = "\n".join(text_chunks)
#     prompt = f"Answer the question based on the context below:\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

#     # --- Generate LVLM answer ---
#     response = ollama.chat(
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}]
#     )

#     print("\n=== LVLM Output ===")
#     print(response['content'])

# # =============================
# # MAIN
# # =============================
# if __name__ == "__main__":
#     query = "Does this laptop have a touch screen? If yes, then how is it?"
#     ask_omnivlm(query)

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import os
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
import open_clip
import faiss

# =============================
# CONFIG
# =============================
DEVICE = "cpu"  # CPU-only
TOP_K_TEXT = 3
TOP_K_IMAGE = 3
RESULTS_FOLDER = "results_images"
GEMMA_MODEL_NAME = "google/gemma-3-4b-it"

# =============================
# LOAD TEXT + IMAGE MODELS
# =============================
print("Loading text and image models...")
text_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load FAISS indexes and metadata
TEXT_INDEX_FILE = "text_index.faiss"
IMAGE_INDEX_FILE = "image_index.faiss"
TEXT_METADATA_FILE = "text_metadata.npy"
IMAGE_METADATA_FILE = "image_metadata.npy"

text_index = faiss.read_index(TEXT_INDEX_FILE)
image_index = faiss.read_index(IMAGE_INDEX_FILE)
text_metadata = np.load(TEXT_METADATA_FILE, allow_pickle=True)
image_metadata = np.load(IMAGE_METADATA_FILE, allow_pickle=True)

# =============================
# LOAD GEMMA 3 MODEL (CPU)
# =============================
print("Loading Gemma 3 model on CPU...")
gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
    GEMMA_MODEL_NAME,
    torch_dtype=torch.float32,
    use_auth_token=True
).eval().to(DEVICE)

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_NAME)

# =============================
# CONTEXT RETRIEVAL
# =============================
def retrieve_context(query, top_k_text=TOP_K_TEXT, top_k_image=TOP_K_IMAGE):
    # --- Text ---
    q_emb_text = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D_txt, I_txt = text_index.search(q_emb_text, top_k_text)
    text_chunks = [text_metadata[idx]["content"] for idx in I_txt[0]]

    # --- Images ---
    tokens = tokenizer(query).to(DEVICE)
    with torch.no_grad():
        q_emb_img = clip_model.encode_text(tokens)
        q_emb_img /= q_emb_img.norm(dim=-1, keepdim=True)
    D_img, I_img = image_index.search(q_emb_img.cpu().numpy(), top_k_image)

    # Prepare results folder
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    for f in os.listdir(RESULTS_FOLDER):
        os.remove(os.path.join(RESULTS_FOLDER, f))

    retrieved_images = []
    for rank, idx in enumerate(I_img[0]):
        meta = image_metadata[idx]
        if meta["type"] == "image":
            src_path = os.path.join("key_frames", meta["file"])
            if os.path.exists(src_path):
                dst_path = os.path.join(RESULTS_FOLDER, f"rank{rank+1}_{meta['file']}")
                shutil.copy(src_path, dst_path)
                retrieved_images.append(dst_path)

    return text_chunks, retrieved_images

# =============================
# ASK GEMMA 3
# =============================
def ask_gemma(query):
    text_chunks, images = retrieve_context(query)

    print("\n=== Retrieved Text Chunks ===")
    for i, chunk in enumerate(text_chunks, 1):
        print(f"{i}. {chunk}\n")

    print("=== Retrieved Images ===")
    for img in images:
        print(img)

    context_text = "\n".join(text_chunks)

    # 🧩 Build multimodal conversation for Gemma 3
    messages = []
    content = []
    for img_path in images:
        content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
    content.append({"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"})

    messages.append({"role": "user", "content": content})

    # 👇 Use apply_chat_template — this automatically handles image tokens
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=text_prompt,
        images=[Image.open(img).convert("RGB") for img in images] if images else None,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = gemma_model.generate(**inputs, max_new_tokens=200)
        decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

    print("\n=== GEMMA 3 Output ===")
    print(decoded)



# =============================
# MAIN
# =============================
if __name__ == "__main__":
    query = "Does this laptop have a touch screen? If yes, then how is it?"
    ask_gemma(query)
