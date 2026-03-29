import os
from src.database.schema import init_db
from src.database.db_manager import DatabaseManager
from src.ingestion.pubmed_scraper import PubMedScraper
from src.ingestion.case_ingestor import ClinicalCaseIngestor

def run_ingestion_pipeline():
    print("[*] Starting MedRAG-X Data Ingestion Pipeline...")
    
    # 1. Initialize Neon DB Schema directly via SQLAlchemy
    # db_path logic is deprecated as Neon is remote
    print("[+] Initializing Neon DB Schema...")
    init_db()
    
    db_manager = DatabaseManager()
    
    # 2. PubMed / Literature Scraper Pipeline
    print("[+] Running PubMed Ingestion...")
    pubmed = PubMedScraper() # Pass NCBI_API_KEY if available via os.environ.get
    
    # Scraping a sample query relevant to our focus
    search_queries = ["Anemia clinical guidelines", "Chest X-Ray Pneumonia"]
    
    for query in search_queries:
        articles = pubmed.run_pipeline(query, max_results=2)
        for article in articles:
            db_manager.insert_article(
                article_id=article["id"],
                content=article["content"],
                source=article["source"]
            )
            print(f"    -> Inserted Article: {article['id']}")

    # 3. Clinical Case Database Ingestion (MIMIC-IV structure)
    print("\n[+] Running Clinical Case Ingestion...")
    case_ingestor = ClinicalCaseIngestor()
    cases = case_ingestor.load_mock_cases()
    
    for case in cases:
        # Route to cases table
        db_manager.insert_case(
            case_id=case["case_id"],
            features=case["patient_features"],
            diagnosis=case["primary_diagnosis"]
        )
        print(f"    -> Inserted Case: {case['case_id']}")
        
        # Extract Graph Edges securely into Knowledge Graph table
        edges = case_ingestor.extract_graph_edges(case)
        for edge in edges:
            db_manager.insert_graph_edge(
                source_entity=edge["source"],
                relation=edge["relation"],
                target_entity=edge["target"]
            )
            print(f"       -> Added Graph Edge: ({edge['source']}) -> [{edge['relation']}] -> ({edge['target']})")

    print("\n[*] Ingestion Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_ingestion_pipeline()
