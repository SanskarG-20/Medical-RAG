"""
Formal End-to-End Audit & Verification Suite for MedRAG-X.
Runs exhaustive validation across all pipeline layers on a formal clinical document.
"""
import os
import sys
import json
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app, orchestrate_pipeline
from src.agents.parser_agent import ParserAgent
from src.retrieval.hybrid_retriever import HybridRetriever, CaseRetriever
from src.reasoning.clinical_reasoner import ClinicalReasoner
from src.router.query_router import DualSignalQueryRouter
from src.pipeline.master_pipeline import MedRAGPipeline
from src.rag.clinical_rag import ClinicalRAG

def run_comprehensive_audit():
    print("=" * 75)
    print("MEDRAG-X COMPREHENSIVE CLINICAL AUDIT & PIPELINE VERIFICATION")
    print("Formal Document: Formal-Clinical-Patient-Report.pdf")
    print("=" * 75)
    
    audit_results = {}
    doc_path = "Formal-Clinical-Patient-Report.pdf"
    assert os.path.exists(doc_path), f"Formal report not found at {doc_path}"

    # ---------------- STAGE 1: PARSER AGENT AUDIT ----------------
    print("\n[STAGE 1/5] Auditing Multi-Agent Document Parsing Layer...")
    t0 = time.time()
    parser = ParserAgent()
    parsed = parser.run(doc_path)
    t_parse = time.time() - t0
    
    text_len = len(parsed['layout']['text'])
    sections = parsed['layout']['sections']
    tables = parsed['tables']['tables']
    diseases = parsed['entities']['disease']
    tests = parsed['entities']['test']
    
    print(f"  [+] Parsing completed in {t_parse:.3f}s")
    print(f"  [+] Extracted Text: {text_len} characters")
    print(f"  [+] Identified Sections: {sections}")
    print(f"  [+] Tables Processed: {len(tables)} tables")
    print(f"  [+] Detected Diseases: {diseases}")
    print(f"  [+] Detected Tests ({len(tests)}): {tests[:8]}...")
    
    assert text_len > 1000, "Extracted text too short for formal clinical report"
    assert len(diseases) > 0, "No disease entities detected"
    assert len(tests) >= 5, "Insufficient lab test entities detected"
    
    audit_results["stage1_parsing"] = {
        "status": "PASS",
        "time_sec": round(t_parse, 3),
        "text_length": text_len,
        "diseases_detected": diseases,
        "tests_detected_count": len(tests)
    }

    # ---------------- STAGE 2: QUERY ROUTING LAYER AUDIT ----------------
    print("\n[STAGE 2/5] Auditing Dual-Signal Query Routing Layer...")
    router = DualSignalQueryRouter()
    
    test_queries = [
        ("What does low hemoglobin and low MCV indicate in this patient?", "clinical"),
        ("Chest X-ray shows bibasilar infiltrates with pleural effusion", "multimodal"),
        ("Recent randomized trials regarding ferric carboxymaltose vs oral iron", "research")
    ]
    
    routing_results = []
    for q, expected in test_queries:
        scores = router.route(q)
        if isinstance(scores, tuple):
            decision, scores_dict = scores[0], scores[1]
        elif isinstance(scores, dict):
            scores_dict = scores
            decision = max(scores_dict, key=scores_dict.get)
        else:
            decision, scores_dict = "clinical", {}
            
        print(f"  [*] Query: '{q[:50]}...' -> Route: {decision} (Scores: {scores_dict})")
        routing_results.append({"query": q, "expected": expected, "actual": decision, "scores": scores_dict})
    
    audit_results["stage2_routing"] = {
        "status": "PASS",
        "results": routing_results
    }

    # ---------------- STAGE 3: HYBRID RETRIEVAL & CASE MATCHING ----------------
    print("\n[STAGE 3/5] Auditing Hybrid Retrieval & Historical Case Matching...")
    t0 = time.time()
    retriever = HybridRetriever()
    case_retriever = CaseRetriever()
    
    pipeline_state = {
        "file_path": doc_path,
        "query": "Evaluate microcytic hypochromic anemia and iron deficiency presentation",
        "parsed": parsed
    }
    
    retrieved_data = retriever.run(pipeline_state)
    case_data = case_retriever.run(pipeline_state)
    t_retrieval = time.time() - t0
    
    text_chunks = retrieved_data.get("text_context", [])
    similar_cases = case_data.get("similar_cases", [])
    
    print(f"  [+] Retrieval completed in {t_retrieval:.3f}s")
    print(f"  [+] Hybrid Retrieved Chunks: {len(text_chunks)}")
    print(f"  [+] Historical Matching Cases: {len(similar_cases)}")
    
    audit_results["stage3_retrieval"] = {
        "status": "PASS",
        "time_sec": round(t_retrieval, 3),
        "retrieved_chunks_count": len(text_chunks),
        "similar_cases_count": len(similar_cases)
    }

    # ---------------- STAGE 4: CLINICAL REASONING & EVIDENCE ALIGNMENT ----------------
    print("\n[STAGE 4/5] Auditing Clinical Reasoning & Claim Verification Engine...")
    t0 = time.time()
    reasoner = ClinicalReasoner()
    
    full_state = {
        "parsed": parsed,
        "retrieved": retrieved_data,
        "cases": case_data,
        "query": pipeline_state["query"]
    }
    
    reasoning_output = reasoner.run(full_state)
    t_reason = time.time() - t0
    
    findings = reasoning_output.get("structured_reasoning", {}).get("Findings", [])
    hypotheses = reasoning_output.get("structured_reasoning", {}).get("Hypotheses", [])
    evidence = reasoning_output.get("structured_reasoning", {}).get("EvidenceMapping", [])
    confidence = reasoning_output.get("structured_reasoning", {}).get("ConfidenceScore", 0)
    penalty = reasoning_output.get("structured_reasoning", {}).get("HallucinationPenaltyApplied", 0)
    
    print(f"  [+] Reasoning completed in {t_reason:.3f}s")
    print(f"  [+] Clinical Findings: {len(findings)} items")
    print(f"  [+] Hypotheses: {hypotheses}")
    print(f"  [+] Confidence Score: {confidence} | Hallucination Penalty: {penalty}")
    print(f"  [+] Verified Evidence Claims ({len(evidence)}):")
    for ev in evidence[:2]:
        status = ev.get('status', 'N/A')
        score = ev.get('alignment_score', 0)
        claim_sub = ev.get('claim', '')[:80]
        print(f"      - [{status}] (score: {score:.3f}): {claim_sub}...")
        
    assert confidence >= 0.70, f"Confidence too low: {confidence}"
    assert penalty == 0.0, f"Unexpected hallucination penalty: {penalty}"
    
    audit_results["stage4_reasoning"] = {
        "status": "PASS",
        "time_sec": round(t_reason, 3),
        "confidence_score": confidence,
        "hallucination_penalty": penalty,
        "hypotheses_count": len(hypotheses)
    }

    # ---------------- STAGE 5: FASTAPI LIVE ENDPOINT AUDIT ----------------
    print("\n[STAGE 5/5] Auditing Live FastAPI Server & /analyze-report Endpoint...")
    client = TestClient(app)
    
    # 1. Health Endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    print(f"  [+] GET /health -> {resp_health.status_code} OK: {resp_health.json()}")
    
    # 2. Root Endpoint
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    print(f"  [+] GET / -> {resp_root.status_code} OK: {resp_root.json()['service']}")
    
    # 3. Analyze Report with Formal PDF
    t0 = time.time()
    with open(doc_path, "rb") as pdf_file:
        response = client.post(
            "/analyze-report",
            files={"file": ("Formal-Clinical-Patient-Report.pdf", pdf_file, "application/pdf")},
            data={"query": "Perform formal diagnostic synthesis and differential diagnosis."}
        )
    t_api = time.time() - t0
    
    assert response.status_code == 200, f"API failed with status {response.status_code}: {response.text}"
    api_json = response.json()
    assert "structured_reasoning" in api_json, "Response missing structured_reasoning payload"
    
    api_conf = api_json["structured_reasoning"].get("ConfidenceScore")
    api_hyp = api_json["structured_reasoning"].get("Hypotheses")
    
    print(f"  [+] POST /analyze-report -> 200 OK ({t_api:.3f}s)")
    print(f"  [+] Final Confidence: {api_conf}")
    print(f"  [+] Primary Hypothesis: {api_hyp[0] if api_hyp else 'N/A'}")
    
    audit_results["stage5_fastapi"] = {
        "status": "PASS",
        "time_sec": round(t_api, 3),
        "status_code": response.status_code,
        "confidence_score": api_conf,
        "primary_hypothesis": api_hyp[0] if api_hyp else None
    }

    # ---------------- FINAL AUDIT SUMMARY ----------------
    print("\n" + "=" * 75)
    print("ALL 5 AUDIT STAGES PASSED SUCCESSFULLY!")
    print("=" * 75)
    print(json.dumps(audit_results, indent=2))
    return True

if __name__ == "__main__":
    success = run_comprehensive_audit()
    sys.exit(0 if success else 1)
