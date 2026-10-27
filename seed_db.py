import os
import sys

# Ensure this script runs from the project root
sys.path.append(os.getcwd())

from src.database.schema import init_db
from src.database.db_manager import DatabaseManager

def seed_database():
    print("Initializing Database Tables...")
    init_db()
    
    print("Connecting to DB Manager...")
    db = DatabaseManager()
    
    print("Inserting mock articles...")
    db.insert_article(
        article_id="PMID:334567", 
        content="Iron deficiency anemia is the most common form of anemia. Patients present with fatigue, shortness of breath, and low hemoglobin levels. Treatment involves oral iron supplementation (e.g., ferrous sulfate) and identifying the underlying source of blood loss.", 
        source="PubMed"
    )
    db.insert_article(
        article_id="PMID:334568", 
        content="Pneumonia often presents with fever, cough, and lobar consolidation on chest X-rays. Bacterial pneumonia is typically treated with antibiotics like amoxicillin or macrolides.", 
        source="PubMed"
    )

    print("Inserting mock cases...")
    db.insert_case(
        case_id="Case_P1029",
        features={"hemoglobin": 7.2, "shortness_of_breath": True, "fatigue": True, "age": 45, "gender": "F"},
        diagnosis="Severe Iron Deficiency Anemia"
    )
    db.insert_case(
        case_id="Case_P1030",
        features={"wbc_count": 15000, "fever": True, "cough": True, "xray": "lobar infiltrate", "age": 60, "gender": "M"},
        diagnosis="Bacterial Pneumonia"
    )

    print("Inserting mock graph edges (symptoms -> diseases)...")
    db.insert_graph_edge(source_entity="Fatigue", relation="symptom_of", target_entity="Anemia")
    db.insert_graph_edge(source_entity="Low Hemoglobin", relation="indicator_of", target_entity="Anemia")
    db.insert_graph_edge(source_entity="Fever", relation="symptom_of", target_entity="Pneumonia")
    db.insert_graph_edge(source_entity="Lobar Consolidation", relation="visual_finding_of", target_entity="Pneumonia")

    print("\n[+] Database Seeded Successfully with test data! You can verify Neon is working.")

if __name__ == "__main__":
    seed_database()
