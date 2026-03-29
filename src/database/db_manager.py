import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self, db_path=None):
        # We ignore db_path as Neon is cloud Postgres
        db_url = os.getenv("DATABASE_URL", "sqlite:///medragx.db")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
        self.engine = create_engine(db_url)

    def insert_article(self, article_id: str, content: str, source: str):
        """Routes cleaned article text to Neon DB (UPSERT pattern)."""
        with self.engine.connect() as conn:
            query = text("""
                INSERT INTO articles (id, content, source) 
                VALUES (:id, :content, :source)
                ON CONFLICT (id) DO UPDATE 
                SET content = EXCLUDED.content, source = EXCLUDED.source
            """)
            conn.execute(query, {"id": article_id, "content": content, "source": source})
            conn.commit()

    def insert_case(self, case_id: str, features: dict, diagnosis: str):
        """Routes structured patient historical cases to Neon DB."""
        with self.engine.connect() as conn:
            query = text("""
                INSERT INTO cases (id, features, diagnosis) 
                VALUES (:id, :features, :diagnosis)
                ON CONFLICT (id) DO UPDATE 
                SET features = EXCLUDED.features, diagnosis = EXCLUDED.diagnosis
            """)
            conn.execute(query, {"id": case_id, "features": json.dumps(features), "diagnosis": diagnosis})
            conn.commit()

    def insert_graph_edge(self, source_entity: str, relation: str, target_entity: str):
        """Routes extracted entity relationships to Knowledge Graph edge table."""
        with self.engine.connect() as conn:
            query = text("""
                INSERT INTO graph_edges (source, relation, target) 
                VALUES (:source, :relation, :target)
            """)
            conn.execute(query, {"source": source_entity, "relation": relation, "target": target_entity})
            conn.commit()
        
    def insert_image_metadata(self, img_id: str, embedding: bytes, metadata: dict):
        """Routes binary image embeddings to Neon BYTEA columns."""
        with self.engine.connect() as conn:
            query = text("""
                INSERT INTO images (id, embedding, metadata) 
                VALUES (:id, :embedding, :metadata)
                ON CONFLICT (id) DO UPDATE 
                SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
            """)
            conn.execute(query, {"id": img_id, "embedding": embedding, "metadata": json.dumps(metadata)})
            conn.commit()