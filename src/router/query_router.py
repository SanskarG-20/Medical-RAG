try:
    import spacy
except ImportError:
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import numpy as np

class DualSignalQueryRouter:
    def __init__(self):
        # 1. Entity Grounding: SciSpacy for UMLS Extraction
        if spacy:
            try:
                self.nlp = spacy.load("en_core_sci_md")
            except OSError:
                print("spacy model not found, falling back")
                self.nlp = None
        else:
            self.nlp = None
            
        # 2. Intent Classifier: MiniLM to map semantics to analytical vs perceptual vs research intents
        if SentenceTransformer:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.anchors = {
                "clinical": self.encoder.encode("patient history symptoms lab values vitals admission notes"),
                "multimodal": self.encoder.encode("x-ray mri scan radiograph opacity image visual findings"),
                "research": self.encoder.encode("clinical trials standard of care latest treatment guidelines study papers")
            }
        else:
            self.encoder = None
            self.anchors = {
                "clinical": np.random.rand(384),
                "multimodal": np.random.rand(384),
                "research": np.random.rand(384)
            }

        # UMLS / Clinical keyword dictionaries
        self.multimodal_keywords = {"radiography", "mri", "scan", "x-ray", "xray", "ultrasound", "tomography", "imaging", "ct scan", "radiograph", "opacity"}
        self.clinical_keywords = {
            "hemoglobin", "hba1c", "vitals", "fatigue", "fever", "pain", "blood", "patient",
            "anemia", "platelet", "wbc", "rbc", "diagnosis", "symptoms", "history", "shortness of breath",
            "chest pain", "pulse", "bp", "tachycardia", "hypotension", "lab", "labs", "condition"
        }
        self.research_keywords = {
            "guideline", "guidelines", "trial", "trials", "study", "studies", "literature",
            "pubmed", "rct", "meta-analysis", "standard of care", "efficacy", "cohort", "evidence"
        }

    def _cosine_similarity(self, v1, v2):
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.dot(v1, v2) / norm if norm > 1e-10 else 0.0

    def route(self, query: str, has_image: bool = False) -> dict:
        """
        Executes the dual-signal routing:
        1. Fast NER / keyword grounding (clinical, multimodal, research).
        2. Lightweight classifier for intent (MiniLM).
        Returns a calibrated, weighted routing vector.
        """
        routing_vector = {"clinical": 0.0, "multimodal": 0.0, "research": 0.0}
        query_lower = query.lower()

        # --- SIGNAL 1: Entity & UMLS Grounding ---
        detected_imaging = False
        detected_clinical = False
        detected_research = False

        if self.nlp:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    ent_text = ent.text.lower()
                    if any(kw in ent_text for kw in self.multimodal_keywords):
                        detected_imaging = True
                    if any(kw in ent_text for kw in self.clinical_keywords):
                        detected_clinical = True
                    if any(kw in ent_text for kw in self.research_keywords):
                        detected_research = True
            except Exception:
                pass

        # Keyword matching fallback/supplement
        if any(kw in query_lower for kw in self.multimodal_keywords):
            detected_imaging = True
        if any(kw in query_lower for kw in self.clinical_keywords):
            detected_clinical = True
        if any(kw in query_lower for kw in self.research_keywords):
            detected_research = True

        if has_image or detected_imaging:
            routing_vector["multimodal"] += 0.5
        if detected_clinical:
            routing_vector["clinical"] += 0.4
        if detected_research:
            routing_vector["research"] += 0.4
            
        # --- SIGNAL 2: Semantic Intent Classification (MiniLM) ---
        if self.encoder:
            query_vec = self.encoder.encode(query)
        else:
            query_vec = np.random.rand(384)
        
        sim_clinical = max(0.0, float(self._cosine_similarity(query_vec, self.anchors["clinical"])))
        sim_multimodal = max(0.0, float(self._cosine_similarity(query_vec, self.anchors["multimodal"])))
        sim_research = max(0.0, float(self._cosine_similarity(query_vec, self.anchors["research"])))

        # Normalize intent scores
        routing_vector["clinical"] += sim_clinical * 0.8
        routing_vector["multimodal"] += sim_multimodal * 0.8
        routing_vector["research"] += sim_research * 0.8

        # Ensure that if all scores are negligible, the primary detected intent is represented
        max_intent = max(routing_vector.values())
        if max_intent < 0.2:
            routing_vector["clinical"] = 0.5

        # Final Cap and Round
        for k in routing_vector:
            routing_vector[k] = min(1.0, round(routing_vector[k], 2))

        return routing_vector

if __name__ == "__main__":
    router = DualSignalQueryRouter()
    
    print("Test 1: 'What does the latest scan show regarding the pulmonary opacity?'")
    print(router.route("What does the latest scan show regarding the pulmonary opacity?"))
    
    print("\nTest 2: 'Look up the last 3 days of vitals, hemoglobin is dropping.'")
    print(router.route("Look up the last 3 days of vitals, hemoglobin is dropping."))
    
    print("\nTest 3: 'What is the current standard of care for severe iron deficiency anemia?'")
    print(router.route("What is the current standard of care for severe iron deficiency anemia?"))