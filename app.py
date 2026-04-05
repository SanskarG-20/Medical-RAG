import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import uvicorn
import shutil

from src.agents.parser_agent import ParserAgent
from src.retrieval.hybrid_retriever import HybridRetriever, CaseRetriever
from src.reasoning.clinical_reasoner import ClinicalReasoner

app = FastAPI(title="MedRAG-X API: Advanced Multi-Agent Pipeline", version="2.0.0")

# Initialize Pipeline Components
parser = ParserAgent()
retriever = HybridRetriever()
case_retriever = CaseRetriever()
reasoner = ClinicalReasoner()

def orchestrate_pipeline(file_path: str, initial_query: str):
    """
    Agentic Orchestrator acting as a State Machine.
    Similar to LangGraph execution pattern.
    """
    state_machine = {
        "file_path": file_path,
        "query": initial_query,
        "parsed": None,
        "retrieved": None,
        "cases": None,
        "reasoned": None
    }
    
    try:
        # Step 1: Agentic Parse -> Hard Validated Structured Dict
        state_machine["parsed"] = parser.run(state_machine["file_path"])
        
        # Step 2: Hybrid Retrieval (Vector + Graph + Image + CrossEncoder Rerank)
        state_machine["retrieved"] = retriever.run(state_machine)
        
        # Step 3: Similar Case Retrieval (Separate Embedding Logic)
        state_machine["cases"] = case_retriever.run(state_machine)
        
        # Step 4: Step-by-Step Reasoner Pipeline
        state_machine["reasoned"] = reasoner.run(state_machine)
        
        return state_machine["reasoned"]
        
    except Exception as e:
        # State machine gracefully handles failure
        return {"error": str(e), "failed_state": state_machine}

@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...), query: str = "Analyze this report for immediate diagnoses."):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Execute the Orchestrated Workflow
    result = orchestrate_pipeline(file_path, initial_query=query)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
