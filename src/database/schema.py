import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def init_db():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found. Make sure .env is loaded correctly.")
        db_url = 'postgresql://neondb_owner:npg_4NVun9LMyiAE@ep-aged-firefly-a14820vk-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
        
    # Replace handler to psycopg (v3) for modern async/sync support
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Create reports table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                structured_json TEXT
            )
        '''))

        # Create articles table (scraped papers, news)
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                content TEXT,
                source TEXT
            )
        '''))

        # Create embeddings metadata
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                vector_ref TEXT,
                type TEXT
            )
        '''))

        # Create cases table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                features TEXT,
                diagnosis TEXT
            )
        '''))

        # Create graph_edges table for Knowledge Graph Reasoning
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT,
                relation TEXT,
                target TEXT
            )
        '''))

        # Create multimodal images table
        # Note: Postgres uses BYTEA instead of SQLite's BLOB
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                embedding BYTEA,
                metadata TEXT
            )
        '''))

        conn.commit()
    print("Neon DB Tables initialized successfully.")

if __name__ == "__main__":
    init_db()