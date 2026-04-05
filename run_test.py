import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())

print("Loading router...")
from src.router.query_router import DualSignalQueryRouter
print("Router loaded. Init router...")
router = DualSignalQueryRouter()

print("Loading clinical rag...")
from src.rag.clinical_rag import ClinicalRAG
clinical_rag = ClinicalRAG()

print("Loading context fusion...")
from src.retrieval.context_fusion import ContextFusionEngine
fuser = ContextFusionEngine()

print("Loading clinical reasoner...")
from src.reasoning.clinical_reasoner import ClinicalReasoner
reasoner = ClinicalReasoner()

print("Loading pipeline...")
from src.pipeline.master_pipeline import MedRAGPipeline
pipeline = MedRAGPipeline(
    router=router,
    clinical_rag=clinical_rag,
    multimodal_rag=None,
    research_rag=None,
    fuser=fuser,
    reasoner=reasoner
)

print("Running pipeline...")
query = "Patient has low hemoglobin and fatigue. What is the condition?"
result = pipeline.run(query, patient_id="P1029")
import json
print("\n=== PIPELINE RESULT ===")
print(json.dumps(result, indent=2))
