import sqlite3
import networkx as nx
import os
from sqlalchemy import create_engine, text

try:
    from sentence_transformers import CrossEncoder
    # We remove inline raw initialization and import the new deep pipelines
    from src.retrieval.embedding_pipeline import EmbeddingPipeline
    from src.retrieval.multimodal_index import MultimodalIndex
except ImportError:
    CrossEncoder = None
    EmbeddingPipeline = None
    MultimodalIndex = None


class HybridRetriever:
    def __init__(self, db_path="medragx.db"):
        self.db_path = db_path
        
        # Swapped SQLite for Neon PostgreSQL using SQLAlchemy
        db_url = os.getenv("DATABASE_URL", "sqlite:///medragx.db")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
        self.engine = create_engine(db_url)
        
        # 1. Connecting the True Text Pipeline
        self.text_pipeline = EmbeddingPipeline(db_path=db_path)
        self.text_pipeline.load_index()  # Pulls the generated .index and .pkl
        
        # 2. Connecting the True Image Pipeline
        self.image_pipeline = MultimodalIndex()
        self.image_pipeline.load_index()
        
        # 3. Knowledge Graph
        self.graph = nx.Graph()
        
        # 4. Reranker (Cross-encoder to filter out hallucinations)
        if CrossEncoder:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            self.reranker = None
            
    def _vector_search(self, query: str, k: int = 5):
        """Re-routed to use genuine pipeline embeddings mapping text -> text chunks."""
        return self.text_pipeline.search(query, k=k)

    def _image_search(self, text_query: str, k: int = 3):
        """Multimodal CLIP retrieval mapping text to image vectors limit"""
        return self.image_pipeline.search_by_text(text_query, k=k)

    def _graph_search(self, query_entity: str):
        """Graph Traversal for relational context (symptoms <-> diseases)"""
        try:
            return list(nx.single_source_shortest_path_length(self.graph, query_entity, cutoff=2).keys())
        except nx.NetworkXError:
            return []

    def _sql_search(self, query_params: dict):
        """Structured lookup for demographics, specific labs etc"""
        # Connect to Neon DB
        with self.engine.connect() as conn:
            # We simulate finding historical tags matching entities
            res = ["Historical Match SQL context from Neon"]
        return res

    def _rank(self, query: str, context_chunks: list):
        """CRITICAL: Cross-encoder reranking to ensure high relevance and prevent hallucination"""
        if not self.reranker or not context_chunks:
            return context_chunks
            
        pairs = [[query, chunk] for chunk in context_chunks]
        scores = self.reranker.predict(pairs)
        
        # Sort contexts by score (highest first)
        ranked_pairs = sorted(zip(scores, context_chunks), key=lambda x: x[0], reverse=True)
        
        # Hard threshold filter to enforce clinical hallucination boundary
        top_chunks = [chunk for score, chunk in ranked_pairs if score > 0.5]
        return top_chunks

    def run(self, state: dict):
        """Agentic orchestration method"""
        query_text = state.get("query", "patient symptoms and diagnostics")
        extracted_entities = state.get("parsed", {}).get("entities", {}).get("disease", [])
        
        # 1. Text Search directly leveraging the pipeline logic
        vector_results = self._vector_search(query_text)
        
        # 2. Image Search leverages true CLIP multimodal projections
        image_results = self._image_search(query_text)
        
        # 3. Graph search matches specific extracted entities to semantic symptoms
        graph_results = []
        for entity in extracted_entities:
            graph_results.extend(self._graph_search(entity))
            
        # 4. SQL logic retrieves patient boundaries
        sql_results = self._sql_search({"disease": extracted_entities[0] if extracted_entities else None})
        
        # Aggregate raw text contexts
        combined_text_context = vector_results + graph_results + sql_results
        
        # 5. RERANK THE NOISE
        ranked_context = self._rank(query_text, combined_text_context)
        
        # 6. HARD VALIDATION LAYER
        if len(ranked_context) == 0 and len(image_results) == 0:
            raise Exception("Retrieval failed — no scientifically grounded context found above threshold for reasoning.")
        
        return {
            "text_context": ranked_context,
            "image_context": image_results,
            "graph_relations": graph_results
        }


class CaseRetriever:
    """Separate engine dedicated to historical case matching"""
    def __init__(self, db_path="medragx.db"):
        self.text_pipeline = EmbeddingPipeline(db_path=db_path)
        self.text_pipeline.load_index()

    def run(self, state: dict):
        query_text = state.get("query", "patient history")
        
        # Leveraged embedding logic
        nearest_cases = self.text_pipeline.search(query_text, k=2)
        if nearest_cases:
            return {"similar_cases": nearest_cases}
        
        return {"similar_cases": []}
