# # search_clip_vector_db.py

# import torch
# import open_clip
# import faiss
# import numpy as np
# from PIL import Image

# # =============================
# # CONFIGURATION
# # =============================
# FAISS_INDEX_FILE = "clip_index.faiss"
# METADATA_FILE = "metadata.npy"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOP_K = 5


# # =============================
# # LOAD CLIP MODEL + INDEX
# # =============================
# print("Loading CLIP model...")
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
# model = model.to(DEVICE).eval()

# print("Loading FAISS index and metadata...")
# index = faiss.read_index(FAISS_INDEX_FILE)
# metadata = np.load(METADATA_FILE, allow_pickle=True)


# # =============================
# # TEXT SEARCH
# # =============================
# # =============================
# # TEXT SEARCH
# # =============================
# def search_by_text(query, top_k=TOP_K):
#     print(f"\n🔍 Searching for text query: '{query}'")
#     tokens = tokenizer(query).to(DEVICE)
#     with torch.no_grad():
#         q_emb = model.encode_text(tokens)
#         q_emb /= q_emb.norm(dim=-1, keepdim=True)
#     D, I = index.search(q_emb.cpu().numpy(), top_k)
#     for rank, idx in enumerate(I[0]):
#         meta = metadata[idx]   # ✅ fixed
#         print(f"{rank+1}. {meta}")


# # =============================
# # IMAGE SEARCH
# # =============================
# def search_by_image(image_path, top_k=TOP_K):
#     print(f"\n🔍 Searching for similar to image: {image_path}")
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         q_emb = model.encode_image(image)
#         q_emb /= q_emb.norm(dim=-1, keepdim=True)
#     D, I = index.search(q_emb.cpu().numpy(), top_k)
#     for rank, idx in enumerate(I[0]):
#         meta = metadata[idx]   # ✅ fixed
#         print(f"{rank+1}. {meta}")

# # =============================
# # ENTRY POINT (EXAMPLES)
# # =============================
# if __name__ == "__main__":
#     # Example 1: Text query
#     search_by_text("quality of touchpad")

#     # Example 2: Image query
#     # search_by_image("query_image.jpg")


# search_hybrid_vector_dbs.py

import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import open_clip
import shutil  
import os

# =============================
# CONFIGURATION
# =============================
TEXT_INDEX_FILE = "text_index.faiss"
IMAGE_INDEX_FILE = "image_index.faiss"
TEXT_METADATA_FILE = "text_metadata.npy"
IMAGE_METADATA_FILE = "image_metadata.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5


# =============================
# LOAD MODELS
# =============================
print("Loading models...")
# text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
text_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(DEVICE).eval()

print("Loading FAISS indexes...")
text_index = faiss.read_index(TEXT_INDEX_FILE)
image_index = faiss.read_index(IMAGE_INDEX_FILE)
text_metadata = np.load(TEXT_METADATA_FILE, allow_pickle=True)
image_metadata = np.load(IMAGE_METADATA_FILE, allow_pickle=True)


# =============================
# TEXT SEARCH (semantic)
# =============================
def search_text(query, top_k=TOP_K):
    print(f"\n🔍 Searching text for: '{query}'")
    q_emb = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = text_index.search(q_emb, top_k)
    for rank, idx in enumerate(I[0]):
        meta = text_metadata[idx]
        print(f"{rank+1}. {meta}")


# =============================
# IMAGE SEARCH (similar images)
# =============================
def search_images_by_text(query, top_k=TOP_K):
    print(f"\n🧠 Searching images by text query: '{query}'")

    # 👇 create/clear results folder
    results_folder = "results_images"
    os.makedirs(results_folder, exist_ok=True)
    for f in os.listdir(results_folder):
        os.remove(os.path.join(results_folder, f))

    tokens = tokenizer(query).to(DEVICE)
    with torch.no_grad():
        q_emb = clip_model.encode_text(tokens)
        q_emb /= q_emb.norm(dim=-1, keepdim=True)
    D, I = image_index.search(q_emb.cpu().numpy(), top_k)

    for rank, idx in enumerate(I[0]):
        meta = image_metadata[idx]
        print(f"{rank+1}. {meta}")

        # 👇 copy retrieved images to results folder
        if meta["type"] == "image":
            src_path = os.path.join("key_frames/", meta["file"])
            if os.path.exists(src_path):
                dst_path = os.path.join(results_folder, f"rank{rank+1}_{meta['file']}")
                shutil.copy(src_path, dst_path)

    print(f"\n✅ Top {top_k} images copied to '{results_folder}' folder.\n")


# =============================
# TEXT → IMAGE SEARCH (CLIP)
# =============================
# def search_images_by_text(query, top_k=TOP_K):
#     print(f"\n🧠 Searching images by text query: '{query}'")
#     tokens = tokenizer(query).to(DEVICE)
#     with torch.no_grad():
#         q_emb = clip_model.encode_text(tokens)
#         q_emb /= q_emb.norm(dim=-1, keepdim=True)
#     D, I = image_index.search(q_emb.cpu().numpy(), top_k)
#     for rank, idx in enumerate(I[0]):
#         meta = image_metadata[idx]
#         print(f"{rank+1}. {meta}")


# =============================
# MAIN EXAMPLES
# =============================
if __name__ == "__main__":
    # 1️⃣ Semantic text search
    search_text("do it have touch screen?")

    # 2️⃣ Find similar images by text
    search_images_by_text("do it have touch screen?")

    # 3️⃣ Find similar images by image
    # search_by_image("query.jpg")
