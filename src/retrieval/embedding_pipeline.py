import sqlite3
import numpy as np
import faiss
import pickle
import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, db_path="medragx.db", store_path="medragx_text_store.pkl", index_path="medragx_text.index"):
        self.db_path = db_path
        self.store_path = store_path
        self.index_path = index_path
        
        # Swapped SQLite for Neon PostgreSQL using SQLAlchemy
        db_url = os.getenv("DATABASE_URL", "sqlite:///medragx.db")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
        self.engine = create_engine(db_url)
        
        self.index = None
        self.text_store = []
        
        # Using a fast powerful local embeddings model instead of OpenAI for offline medical compliance
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 

    def fetch_data(self):
        with self.engine.connect() as conn:
            articles_res = conn.execute(text("SELECT content FROM articles")).fetchall()
            articles = [row[0] for row in articles_res if row[0]]

            cases_res = conn.execute(text("SELECT features FROM cases")).fetchall()
            cases = [row[0] for row in cases_res if row[0]]

        return articles + cases

    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def embed_text(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    def build_index(self):
        data = self.fetch_data()
        
        if not data:
            print("[-] No data found in SQLite DB to embed.")
            return

        vectors = []
        for doc in data:
            chunks = self.chunk_text(doc)

            for chunk in chunks:
                vec = self.embed_text(chunk)
                vectors.append(vec)
                self.text_store.append(chunk)

        if not vectors:
            print("[-] No chunks generated.")
            return

        vectors = np.vstack(vectors).astype(np.float32)

        dim = vectors.shape[1]
        # Inner product perfectly fits normalized embeddings
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        print(f"[+] Indexed {len(vectors)} chunks into FAISS text index.")
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.text_store, f)
        print(f"[+] Successfully saved FAISS index and Text Store.")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, "rb") as f:
                self.text_store = pickle.load(f)
            print("[+] Loaded FAISS text index and Text Store into memory.")
        else:
            print("[-] Index files do not exist.")

    def search(self, query, k=5):
        if not self.index:
            self.load_index()
            
        q_vec = self.embed_text(query)
        q_vec = np.array([q_vec], dtype=np.float32)

        distances, indices = self.index.search(q_vec, k)
        
        if len(indices) == 0 or len(indices[0]) == 0 or indices[0][0] == -1:
             return []

        results = [self.text_store[i] for i in indices[0] if i != -1]
        return results

if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    pipeline.build_index()
    
    # Test retrieval
    print("\n--- Testing Retrieval ---")
    results = pipeline.search("patient having shortness of breath and low hemoglobin")
    for idx, r in enumerate(results):
         print(f"{idx+1}. {r[:100]}...")
