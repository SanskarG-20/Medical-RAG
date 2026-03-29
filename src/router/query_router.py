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

        # UMLS Semantic Type approximation keywords for imaging vs clinical
        self.multimodal_keywords = {"radiography", "mri", "scan", "x-ray", "xray", "ultrasound", "tomography", "imaging"}

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def route(self, query: str, has_image: bool = False) -> dict:
        """
        Executes the dual-signal routing:
        1. Fast NER for hard entities (SciSpacy).
        2. Lightweight classifier for intent (MiniLM).
        Returns a weighted routing vector.
        """
        routing_vector = {"clinical": 0.0, "multimodal": 0.0, "research": 0.0}
        query_lower = query.lower()

        # --- SIGNAL 1: Entity & UMLS Grounding (SciSpacy) ---
        detected_imaging_entity = False
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if any(img_kw in ent.text.lower() for img_kw in self.multimodal_keywords):
                    detected_imaging_entity = True
        else:
            if any(img_kw in query_lower for img_kw in self.multimodal_keywords):
                detected_imaging_entity = True

        if has_image or detected_imaging_entity:
            routing_vector["multimodal"] += 0.5
            
        # --- SIGNAL 2: Semantic Intent Classification (MiniLM) ---
        if self.encoder:
            query_vec = self.encoder.encode(query)
        else:
            # mock vector logic
            query_vec = np.random.rand(384)
        
        sim_clinical = self._cosine_similarity(query_vec, self.anchors["clinical"])
        sim_multimodal = self._cosine_similarity(query_vec, self.anchors["multimodal"])
        sim_research = self._cosine_similarity(query_vec, self.anchors["research"])

        # Normalize intent scores to probabilities via Softmax-like or min-max shifting
        # Using a slight threshold cap as a proxy
        routing_vector["clinical"] += float(max(0, sim_clinical) * 0.8)
        routing_vector["multimodal"] += float(max(0, sim_multimodal) * 0.8)
        routing_vector["research"] += float(max(0, sim_research) * 0.8)

        # Baseline adjustments if explicit keywords dictate a hard lean
        if "guideline" in query_lower or "study" in query_lower or "research" in query_lower:
            routing_vector["research"] += 0.4

        # Final Cap
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