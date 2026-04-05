import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()

import json

print("="*60)
print("MEDRAG-X: Running Full Pipeline")
print("="*60)

file_path = "Test-MedicalPatient Report.pdf"
query = "what are factors that interfere with HbA1C measurement?"

print(f"\n[*] PDF: {file_path}")
print(f"[*] Query: {query}\n")

# Step 1: Parse the PDF
print("--- Step 1: Parsing PDF ---")
from src.agents.parser_agent import ParserAgent
parser = ParserAgent()
parsed = parser.run(file_path)
print(f"  Sections found: {parsed['layout']['sections']}")
print(f"  Text length: {len(parsed['layout']['text'])} chars")
print(f"  Entities: {parsed['entities']}")
print()

# Step 2: Retrieve context using the embedding pipeline
print("--- Step 2: Retrieval (ChromaDB) ---")
from src.retrieval.embedding_pipeline import EmbeddingPipeline
pipeline = EmbeddingPipeline()

# Check if collection is empty, if so build from parsed text
if pipeline.collection.count() == 0:
    print("  [!] ChromaDB collection is empty. Indexing parsed PDF text...")
    chunks = pipeline.chunk_text(parsed['layout']['text'], chunk_size=200)
    if chunks:
        embeddings = [pipeline.embed_text(c) for c in chunks]
        ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
        pipeline.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        print(f"  [+] Indexed {len(chunks)} chunks from PDF")
else:
    print(f"  Collection already has {pipeline.collection.count()} chunks")

results = pipeline.search(query, k=5)
print(f"  Retrieved {len(results)} chunks:")
for i, r in enumerate(results):
    print(f"    {i+1}. {r[:120]}...")
print()

# Step 3: Clinical Reasoning with Groq LLM
print("--- Step 3: Clinical Reasoning (Groq LLM) ---")
from src.reasoning.clinical_reasoner import ClinicalReasoner
reasoner = ClinicalReasoner()

state = {
    "parsed": parsed,
    "retrieved": {
        "context": [{"content": chunk, "source_type": "clinical_notes", "evidence_level": "observational"} for chunk in results]
    },
    "query": query
}

result = reasoner.run(state)
print("\n" + "="*60)
print("FINAL ANSWER")
print("="*60)
print(json.dumps(result, indent=2))
