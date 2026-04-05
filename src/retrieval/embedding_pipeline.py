import sqlite3
import numpy as np
import os
import chromadb
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, db_path="medragx.db", store_path="./medrag_chroma", index_path="medragx_text"):
        self.db_path = db_path
        self.store_path = store_path
        self.index_path = index_path
        
        # Swapped SQLite for Neon PostgreSQL using SQLAlchemy
        db_url = os.getenv("DATABASE_URL", "sqlite:///medragx.db")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
        self.engine = create_engine(db_url)
        
        self.chroma_client = chromadb.PersistentClient(path=self.store_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.index_path)
        
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
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def build_index(self):
        data = self.fetch_data()
        
        if not data:
            print("[-] No data found in SQLite DB to embed.")
            return

        documents = []
        embeddings = []
        ids = []
        
        chunk_idx = 0
        for doc in data:
            chunks = self.chunk_text(doc)

            for chunk in chunks:
                vec = self.embed_text(chunk)
                
                documents.append(chunk)
                embeddings.append(vec)
                ids.append(f"doc_{chunk_idx}")
                chunk_idx += 1

        if not documents:
            print("[-] No chunks generated.")
            return

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )

        print(f"[+] Indexed {len(documents)} chunks into ChromaDB text index.")

    def search(self, query, k=5):
        q_vec = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []

        return results['documents'][0]

if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    pipeline.build_index()
    
    # Test retrieval
    print("\n--- Testing Retrieval ---")
    results = pipeline.search("patient having shortness of breath and low hemoglobin")
    for idx, r in enumerate(results):
         print(f"{idx+1}. {r[:100]}...")
