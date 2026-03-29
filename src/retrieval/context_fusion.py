import math
import json
from typing import List, Dict, Any

class ContextFusionEngine:
    def __init__(self, k: int = 60, time_decay_lambda: float = 0.05):
        self.k = k
        self.decay_lambda = time_decay_lambda
        
        self.authority_weights = {
            "multimodal_scan": 0.9,
            "clinical_notes": 0.7,
            "research": 0.4
        }
        self.evidence_modifiers = {
            "meta-analysis": 1.2,
            "RCT": 1.1,
            "observational": 1.0,
            "case report": 0.8
        }
        # Quotas do not apply to hard_facts layer anymore
        self.quotas = {
            "multimodal_scan": 2,
            "clinical_notes": 3,
            "research": 2
        }
        self.order_hierarchy = {
            "sql_patient": 1,
            "multimodal_scan": 2,
            "clinical_notes": 3,
            "research": 4
        }

    def score_chunk(self, chunk: Dict[str, Any], current_time: float) -> float:
        source_type = chunk.get("source_type", "research")
        rank = chunk.get("rank", 10)
        rerank_score = chunk.get("rerank_score", 1.0)
        timestamp = chunk.get("timestamp")
        evidence_level = chunk.get("evidence_level")

        auth_weight = self.authority_weights.get(source_type, 0.4)
        base_rrf = 1.0 / (self.k + rank)
        ev_mod = self.evidence_modifiers.get(evidence_level, 1.0) if evidence_level else 1.0

        temp_weight = 1.0
        if timestamp is not None and source_type == "clinical_notes":
            time_diff_days = max(0, current_time - timestamp)
            temp_weight = math.exp(-self.decay_lambda * time_diff_days)

        return auth_weight * base_rrf * rerank_score * ev_mod * temp_weight

    def detect_conflicts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        text_corpus = " ".join(c.get("content", "").lower() for c in chunks)
        conflicts = []
        if "hb is low" in text_corpus and "hb is normal" in text_corpus:
            conflicts.append("Contradiction: Context contains both 'low' and 'normal' hemoglobin claims.")
        return conflicts

    def fuse(self, retrieved_contexts: Dict[str, Any], current_time: float = 0.0) -> Dict[str, Any]:
        """
        Modified to accept separated hard_facts vs probabilistic queries.
        """
        hard_facts = []
        probabilistic_chunks = []
        
        if "clinical" in retrieved_contexts:
            hard_facts.extend(retrieved_contexts["clinical"].get("hard_facts", []))
            probabilistic_chunks.extend(retrieved_contexts["clinical"].get("narratives", []))
            
        probabilistic_chunks.extend(retrieved_contexts.get("multimodal", []))
        probabilistic_chunks.extend(retrieved_contexts.get("research", []))

        for chunk in probabilistic_chunks:
            chunk["final_score"] = self.score_chunk(chunk, current_time)

        sorted_chunks = sorted(probabilistic_chunks, key=lambda x: x.get("final_score", 0), reverse=True)

        selected_chunks = list(hard_facts) # Hard facts bypass all quotas
        counts = {k: 0 for k in self.quotas.keys()}
        
        for chunk in sorted_chunks:
            src = chunk.get("source_type", "research")
            if counts.get(src, 0) < self.quotas.get(src, 0):
                selected_chunks.append(chunk)
                counts[src] = counts.get(src, 0) + 1

        ordered_chunks = sorted(
            selected_chunks, 
            key=lambda x: self.order_hierarchy.get(x.get("source_type", "research"), 5)
        )

        conflicts = self.detect_conflicts(ordered_chunks)

        return {
            "context": ordered_chunks,
            "conflicts": conflicts
        }

if __name__ == "__main__":
    fusion = ContextFusionEngine()
    
    test_payload = {
        "clinical": {
            "hard_facts": [
                {
                    "content": "Hb is 7.2",
                    "source_type": "sql_patient",
                    "timestamp": 1,
                    "rank": 1
                }
            ],
            "narratives": [
                {
                    "content": "Patient pale.",
                    "source_type": "clinical_notes",
                    "timestamp": 31,
                    "rank": 1
                }
            ]
        }
    }
    
    output = fusion.fuse(test_payload, current_time=2.0)
    print(json.dumps(output, indent=2))
