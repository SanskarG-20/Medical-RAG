from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import faiss
import numpy as np
import os
import pickle

class MultimodalIndex:
    def __init__(self, index_path="medragx_image.index", store_path="medragx_image_store.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CLIP converts images to vectors in the same latent space as it does text.
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.index_path = index_path
        self.store_path = store_path
        self.index = None
        self.image_store = [] # stores paths or IDs representing the image

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Normalize for cosine similarity
            outputs /= outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().numpy()[0]
        
    def embed_text_query(self, text_query):
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Normalize for cosine similarity
            outputs /= outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().numpy()[0]

    def build_index(self, image_paths):
        """Pass a list of valid image paths from data processing layer"""
        if not image_paths:
            print("[-] No images provided to MultimodalIndex.")
            return
            
        vectors = []
        for img_path in image_paths:
            try:
                vec = self.embed_image(img_path)
                vectors.append(vec)
                self.image_store.append(img_path)
            except Exception as e:
                print(f"Failed to embed {img_path}: {str(e)}")

        if not vectors:
            return

        vectors = np.vstack(vectors).astype(np.float32)
        dim = vectors.shape[1]
        
        # Inner Product for normalized embeddings
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        print(f"[+] Indexed {len(vectors)} images into FAISS multimodal index.")
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.image_store, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, "rb") as f:
                self.image_store = pickle.load(f)

    def search_by_text(self, text_query, k=3):
        if not self.index:
            self.load_index()
            if not self.index:
                return []
                
        q_vec = self.embed_text_query(text_query)
        q_vec = np.array([q_vec], dtype=np.float32)

        distances, indices = self.index.search(q_vec, k)
        
        results = [self.image_store[i] for i in indices[0] if i != -1]
        return results

if __name__ == "__main__":
    mm_index = MultimodalIndex()
    # Mocking execution to prevent crashing if user has no images yet
    print("[*] Multimodal Indexer Instantiated. (Provide list of paths to build()).")
