import csv
import os

class ClinicalCaseIngestor:
    """
    Ingests standardized clinical cases (e.g. MIMIC-IV exports) 
    and transforms them into the graph / vector schema requirements.
    """
    def __init__(self, raw_data_path="data/raw/"):
        self.raw_data_path = raw_data_path

    def load_mock_cases(self) -> list:
        """
        Creates mock MIMIC-IV case structural extractions.
        In reality, this would pandas.read_csv() over MIMIC 'admissions', 'diagnoses_icd' tables.
        """
        cases = [
            {
                "case_id": "MIMIC_1001",
                "patient_features": {
                    "age": 45,
                    "gender": "M",
                    "labs": {"hemoglobin": 8.5, "wbc": 14.2, "platelets": 150},
                    "symptoms": ["fatigue", "shortness of breath", "pallor"]
                },
                "primary_diagnosis": "Severe Iron Deficiency Anemia"
            },
            {
                "case_id": "MIMIC_1002",
                "patient_features": {
                    "age": 62,
                    "gender": "F",
                    "labs": {"hemoglobin": 11.2, "wbc": 4.5, "platelets": 250},
                    "symptoms": ["chest pain", "cough", "fever"]
                },
                "primary_diagnosis": "Pneumonia"
            }
        ]
        return cases

    def extract_graph_edges(self, case: dict):
        """Converts case symptomatology into Graph Edges for the Knowledge Graph."""
        edges = []
        diagnosis = case["primary_diagnosis"]
        symptoms = case["patient_features"].get("symptoms", [])
        
        for symptom in symptoms:
            # Add Knowledge Graph Node relations
            edges.append({
                "source": symptom,
                "relation": "INDICATES",
                "target": diagnosis
            })
            
        return edges
