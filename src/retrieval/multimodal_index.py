from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
import chromadb

class MultimodalIndex:
    def __init__(self, index_path="medragx_image", store_path="./medrag_chroma"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self.index_path = index_path
        self.store_path = store_path
        
        self.chroma_client = chromadb.PersistentClient(path=self.store_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.index_path)

    def _ensure_model(self):
        """Lazy load CLIP model and processor on first use."""
        if self._model is None or self._processor is None:
            try:
                self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                print(f"[*] MultimodalIndex: CLIP model unavailable ({e}). Continuing in text-only mode.")
                self._model = None
                self._processor = None

    def embed_image(self, image_path):
        self._ensure_model()
        if not self._model or not self._processor:
            return [0.0] * 512
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            # Normalize for cosine similarity
            outputs /= outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().numpy()[0].tolist()
        
    def embed_text_query(self, text_query):
        self._ensure_model()
        if not self._model or not self._processor:
            return None
        inputs = self._processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
            # Normalize for cosine similarity
            outputs /= outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().numpy()[0].tolist()

    def build_index(self, image_paths):
        """Pass a list of valid image paths from data processing layer"""
        if not image_paths:
            print("[-] No images provided to MultimodalIndex.")
            return
            
        embeddings = []
        metadatas = []
        ids = []
        for i, img_path in enumerate(image_paths):
            try:
                vec = self.embed_image(img_path)
                embeddings.append(vec)
                meta = {"image_path": img_path}
                metadatas.append(meta)
                img_id = f"img_{i}_{os.path.basename(img_path)}"
                ids.append(img_id)
                try:
                    import numpy as np
                    from src.database.db_manager import DatabaseManager
                    DatabaseManager().insert_image_metadata(
                        img_id=img_id,
                        embedding=np.array(vec, dtype=np.float32).tobytes(),
                        metadata=meta
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to embed {img_path}: {str(e)}")

        if not embeddings:
            return

        self.collection.upsert(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"[+] Indexed {len(embeddings)} images into ChromaDB multimodal index.")

    def search_by_text(self, text_query, k=3):
        if self.collection.count() == 0:
            return []
        q_vec = self.embed_text_query(text_query)
        if q_vec is None:
            return []
        
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            return []
            
        image_paths = [meta["image_path"] for meta in results['metadatas'][0]]
        return image_paths

if __name__ == "__main__":
    mm_index = MultimodalIndex()
    # Mocking execution to prevent crashing if user has no images yet
    print("[*] Multimodal Indexer Instantiated. (Provide list of paths to build()).")
