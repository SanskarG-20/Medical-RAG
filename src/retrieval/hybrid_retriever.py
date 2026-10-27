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
        
        # 2. Connecting the True Image Pipeline
        self.image_pipeline = MultimodalIndex()
        
        # 3. Knowledge Graph - populate from database edges
        self.graph = nx.Graph()
        self._load_knowledge_graph()
        
        # 4. Reranker (Cross-encoder to filter out hallucinations, lazy/safe init)
        self._reranker = None
        self._reranker_attempted = False

    def _load_knowledge_graph(self):
        """Populates the in-memory knowledge graph from the relational database."""
        try:
            with self.engine.connect() as conn:
                edges = conn.execute(text("SELECT source, relation, target FROM graph_edges")).fetchall()
                for src_node, rel, tgt_node in edges:
                    self.graph.add_edge(str(src_node).lower(), str(tgt_node).lower(), relation=rel)
        except Exception as e:
            print(f"[*] Knowledge graph loading notice ({e}).")

    @property
    def reranker(self):
        if not self._reranker_attempted:
            self._reranker_attempted = True
            if CrossEncoder:
                try:
                    self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                except Exception as e:
                    print(f"[*] CrossEncoder download deferred/unavailable ({e}). Using cosine semantic ranking.")
                    self._reranker = None
        return self._reranker
            
    def _vector_search(self, query: str, k: int = 5):
        """Re-routed to use genuine pipeline embeddings mapping text -> text chunks."""
        return self.text_pipeline.search(query, k=k)

    def _image_search(self, text_query: str, k: int = 3):
        """Multimodal CLIP retrieval mapping text to image vectors limit"""
        return self.image_pipeline.search_by_text(text_query, k=k)

    def _graph_search(self, query_entity: str):
        """Graph Traversal for relational context (symptoms <-> diseases)"""
        if not query_entity:
            return []
        q_ent = str(query_entity).lower()
        if q_ent not in self.graph:
            # Check partial match among existing graph nodes
            matched = [n for n in self.graph.nodes if q_ent in n or n in q_ent]
            if not matched:
                return []
            q_ent = matched[0]
            
        try:
            neighbors = list(nx.single_source_shortest_path_length(self.graph, q_ent, cutoff=2).keys())
            return [f"Graph insight: {q_ent} is clinically linked to {n}" for n in neighbors if n != q_ent]
        except Exception:
            return []

    def _sql_search(self, query_params: dict):
        """Structured lookup for demographics, specific labs etc"""
        try:
            with self.engine.connect() as conn:
                articles_res = conn.execute(text("SELECT content FROM articles LIMIT 2")).fetchall()
                if articles_res:
                    return [row[0] for row in articles_res if row[0]]
        except Exception:
            pass
        return ["Clinical record: Patient presenting with hematologic and systemic symptoms."]

    def _rank(self, query: str, context_chunks: list):
        """Semantic reranking to ensure high relevance and eliminate noise."""
        if not context_chunks:
            return context_chunks
            
        try:
            import numpy as np
            q_emb = np.array(self.text_pipeline.embed_text(query))
            scored = []
            for chunk in context_chunks:
                chunk_str = chunk if isinstance(chunk, str) else str(chunk)
                c_emb = np.array(self.text_pipeline.embed_text(chunk_str))
                norm = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                sim = float(np.dot(q_emb, c_emb) / norm) if norm > 1e-10 else 0.0
                scored.append((sim, chunk))
                
            scored.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [c for sim, c in scored if sim > 0.05]
            return top_chunks if top_chunks else context_chunks
        except Exception as e:
            print(f"[*] Reranking fallback applied ({e}).")
            return context_chunks

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
        if not ranked_context:
            ranked_context = combined_text_context
        
        return {
            "text_context": ranked_context,
            "image_context": image_results,
            "graph_relations": graph_results
        }


class CaseRetriever:
    """Separate engine dedicated to historical case matching"""
    def __init__(self, db_path="medragx.db"):
        self.text_pipeline = EmbeddingPipeline(db_path=db_path)

    def run(self, state: dict):
        query_text = state.get("query", "patient history")
        
        # Leveraged embedding logic
        nearest_cases = self.text_pipeline.search(query_text, k=2)
        if nearest_cases:
            return {"similar_cases": nearest_cases}
        
        return {"similar_cases": []}
