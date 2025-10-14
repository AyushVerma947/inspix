# # build_clip_vector_db.py

# import os
# import torch
# import open_clip
# import faiss
# import numpy as np
# from PIL import Image
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # =============================
# # CONFIGURATION
# # =============================
# IMAGE_FOLDER = "key_frames"        # Folder containing your images
# TEXT_FILE = "transcription.txt"     # Text file path
# FAISS_INDEX_FILE = "clip_index.faiss"
# METADATA_FILE = "metadata.npy"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CHUNK_SIZE = 300
# CHUNK_OVERLAP = 50


# # =============================
# # LOAD CLIP MODEL
# # =============================
# print("Loading CLIP model...")
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
# model = model.to(DEVICE).eval()


# # =============================
# # IMAGE EMBEDDINGS
# # =============================
# def embed_images(image_folder):
#     print("Embedding images...")
#     image_embeddings = []
#     image_metadata = []
#     for file in os.listdir(image_folder):
#         if file.lower().endswith((".png", ".jpg", ".jpeg")):
#             path = os.path.join(image_folder, file)
#             image = preprocess(Image.open(path)).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 emb = model.encode_image(image)
#                 emb /= emb.norm(dim=-1, keepdim=True)
#             image_embeddings.append(emb.cpu().numpy())
#             image_metadata.append({"type": "image", "file": file})
#     if image_embeddings:
#         image_embeddings = np.vstack(image_embeddings)
#     else:
#         image_embeddings = np.empty((0, 512), dtype="float32")
#     return image_embeddings, image_metadata


# # =============================
# # TEXT EMBEDDINGS
# # =============================
# def embed_text_chunks(text_path):
#     print("Embedding text chunks...")
#     text = open(text_path, "r", encoding="utf-8").read()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     chunks = splitter.split_text(text)

#     text_embeddings = []
#     text_metadata = []
#     for i, chunk in enumerate(chunks):
#         tokens = tokenizer(chunk).to(DEVICE)
#         with torch.no_grad():
#             emb = model.encode_text(tokens)
#             emb /= emb.norm(dim=-1, keepdim=True)
#         text_embeddings.append(emb.cpu().numpy())
#         text_metadata.append({"type": "text", "chunk_id": i, "content": chunk})

#     if text_embeddings:
#         text_embeddings = np.vstack(text_embeddings)
#     else:
#         text_embeddings = np.empty((0, 512), dtype="float32")
#     return text_embeddings, text_metadata


# # =============================
# # BUILD VECTOR DB
# # =============================
# def build_vector_db():
#     img_embs, img_meta = embed_images(IMAGE_FOLDER)
#     txt_embs, txt_meta = embed_text_chunks(TEXT_FILE)

#     embeddings = np.vstack([img_embs, txt_embs]).astype("float32")
#     metadata = img_meta + txt_meta

#     print(f"Total embeddings: {len(metadata)}")

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, FAISS_INDEX_FILE)

#     np.save(METADATA_FILE, metadata, allow_pickle=True)
#     print(f"✅ Vector DB saved: {FAISS_INDEX_FILE}, metadata: {METADATA_FILE}")


# if __name__ == "__main__":
#     build_vector_db()


# build_hybrid_vector_dbs.py

import os
import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import open_clip
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================
# CONFIGURATION
# =============================
IMAGE_FOLDER = "key_frames"
TEXT_FILE = "transcription.txt"
TEXT_INDEX_FILE = "text_index.faiss"
IMAGE_INDEX_FILE = "image_index.faiss"
TEXT_METADATA_FILE = "text_metadata.npy"
IMAGE_METADATA_FILE = "image_metadata.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


# =============================
# LOAD MODELS
# =============================
print("Loading models...")

# For text → sentence-transformers
# text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

text_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)


# For images → CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()


# =============================
# TEXT EMBEDDINGS
# =============================
def embed_text_chunks(text_path):
    print("Embedding text chunks...")
    text = open(text_path, "r", encoding="utf-8").read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)

    text_embeddings = text_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    metadata = [{"type": "text", "chunk_id": i, "content": chunk} for i, chunk in enumerate(chunks)]

    return np.array(text_embeddings).astype("float32"), metadata


# =============================
# IMAGE EMBEDDINGS
# =============================
def embed_images(image_folder):
    print("Embedding images...")
    image_embeddings = []
    metadata = []

    for file in os.listdir(image_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(image_folder, file)
            image = preprocess(Image.open(path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_image(image)
                emb /= emb.norm(dim=-1, keepdim=True)
            image_embeddings.append(emb.cpu().numpy())
            metadata.append({"type": "image", "file": file})

    if not image_embeddings:
        raise ValueError("No images found in folder!")

    return np.vstack(image_embeddings).astype("float32"), metadata


# =============================
# BUILD FAISS INDEXES
# =============================
def build_faiss_index(embeddings, index_file):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index


# =============================
# MAIN
# =============================
def main():
    # --- Text ---
    text_embs, text_meta = embed_text_chunks(TEXT_FILE)
    build_faiss_index(text_embs, TEXT_INDEX_FILE)
    np.save(TEXT_METADATA_FILE, text_meta, allow_pickle=True)
    print(f"✅ Text index built: {TEXT_INDEX_FILE}")

    # --- Images ---
    img_embs, img_meta = embed_images(IMAGE_FOLDER)
    build_faiss_index(img_embs, IMAGE_INDEX_FILE)
    np.save(IMAGE_METADATA_FILE, img_meta, allow_pickle=True)
    print(f"✅ Image index built: {IMAGE_INDEX_FILE}")


if __name__ == "__main__":
    main()
