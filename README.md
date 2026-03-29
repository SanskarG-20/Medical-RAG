# MedRAG-X: Multimodal Clinical Intelligence System

MedRAG-X is an agentic, multimodal Retrieval-Augmented Generation (RAG) system that understands long medical reports, processes images and text jointly, retrieves clinical knowledge and similar cases, and produces doctor-level reasoning with transparent evidence tracking.

## System Architectures

### 1. Master Architecture
The Master Architecture shows the end-to-end flow of data and requests. It orchestrates the ingestion of data, the agentic parsing pipeline, routing, retrieval, and final reasoning.

```mermaid
graph TD
    %% Master Architecture
    subgraph Data Sources
        P[PubMed] --> Ingestion
        M[MIMIC-IV] --> Ingestion
        N[NIH Chest X-ray] --> Ingestion
    end

    subgraph Agentic Parsing Pipeline
        Ingestion --> O[Orchestrator Agent]
        O --> L[Layout Agent]
        O --> T[Table Agent]
        O --> I[Image Agent]
        O --> E[Entity Agent]
        O -.-> V{Validator Agent}
    end

    subgraph Multimodal RAG & Router
        V -- Validated Data --> DB[(Vector DB / Neon Postgres)]
        Q[User Query] --> R[Dual-Signal Query Router]
        R -->|Routing Weights| Retrieval
        DB --> Retrieval
    end

    subgraph Reasoning Engine
        Retrieval --> CF[Context Fusion Engine]
        CF --> CR[Clinical Reasoner]
        CR --> Out[Evidence-Backed Clinical Output]
    end
```

### 2. Agentic Parsing System
Documents (PDFs, EHRs) are handled by a multi-agent parser acting as a state machine. It correctly extracts structure, tabular data, images, and medical entities, with a hard validation mechanism to prevent hallucinations.

```mermaid
graph LR
    %% Parsing Pipeline
    PDF[Raw Medical PDF] --> Parser[Parser Orchestrator]
    
    Parser --> Layout[Layout Agent\nDetectron2]
    Parser --> Table[Table Agent\nCamelot/LLM]
    Parser --> Image[Image Agent\nCLIP]
    Parser --> Entity[Entity Agent\nSciSpacy]
    
    Layout --> Context[Context Aggregation]
    Table --> Context
    Image --> Context
    Entity --> Context
    
    Context --> Validator{Validator Agent}
    Validator -- Pass --> Structured[Structured Output\nJSON/Dict]
    Validator -- Fail --> Fallback[Fallback/LLM Parsing]
    Fallback --> Validator
```

### 3. Dual-Signal Query Router
The Query Router intelligently determines the intent of a query, splitting weighting between Clinical Facts, Multimodal Images, and Research Literature using a dual-signal process.

```mermaid
sequenceDiagram
    participant User
    participant QRouter as Dual-Signal Router
    participant Spacy as SciSpacy (Entity Grounding)
    participant Model as MiniLM (Intent Classifier)
    
    User->>QRouter: "What does the scan show for opacity?"
    QRouter->>Spacy: Extract Entities (UMLS)
    Spacy-->>QRouter: Semantic Medical Matching
    QRouter->>Model: Compute Semantic Anchors
    Model-->>QRouter: Cosine Similarity Scores
    QRouter-->>User: Routing Weights (e.g. Multimodal: 0.8, Clinical: 0.2)
```

### 4. Hybrid Retrieval System
The Context Fusion engine merges deterministic SQL queries (for hard facts like lab values) with vector-based semantic retrieval and dense graph lookups. It ensures critical patient data isn't lost in probabilistic vector spaces.

```mermaid
graph TD
    %% Hybrid Retrieval
    Router[Query & Routing Weights] --> HR[Hybrid Retriever]
    
    HR --> |Hard Facts| SQL[(SQL / PostgreSQL)\nLab Values, Meds]
    HR --> |Narrative| Vec[(FAISS / Vector DB)\nClinical Notes, Papers]
    HR --> |Visual| Multimodal[(Multimodal DB)\nX-rays, MRIs]
    HR --> |Similar Cases| Graph[(Knowledge Graph)]

    SQL --> Fusion[Context Fusion Engine]
    Vec --> Fusion
    Multimodal --> Fusion
    Graph --> Fusion
    
    Fusion --> |Cross-Encoder Rerank| FinalContext[Ranked Multi-Source Context]
```

### 5. Clinical Reasoning Engine
The step-by-step reasoning pipeline takes the fused context and produces definitive, doctor-level diagnoses, citing specific chunk IDs and evidence levels.

```mermaid
graph TD
    %% Reasoning Engine
    Ctx[Fused Ranked Context] --> R1[Step 1: Clinical Summary Generation]
    R1 --> R2[Step 2: Differential Diagnosis Hypothesis]
    R2 --> R3[Step 3: Evidence Grounding & Confidence Scoring]
    
    R3 --> Output{Final Output}
    Output --> |Diagnosis| D[Diagnosis Text]
    Output --> |Evidence| E[Citations & Source Chunks]
    Output --> |Confidence| C[Confidence Metric High/Med/Low]
```

## Setup & Execution

### Prerequisites
Make sure your environment is configured for Python 3.10+ and you have necessary C++ build tools for `spacy` and `psycopg`.

If you have issues building `spacy` on macOS, consider using `conda` or installing prebuilt binaries.

```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_md
```

### Database Configuration
We use Neon PostgreSQL for scalable deployment. Be sure to configure your environment variables:
```bash
export DATABASE_URL="postgresql://user:password@neondb.aws.neon.tech/dbname"
```
*(If unset, it will fail back gracefully to local SQLite where supported)*

### Running the Application

1. **Start the API Server:**
```bash
python app.py
```
This runs the FastAPI server with the orchestrated state machine on port 8000.

2. **Run Tests:**
```bash
python run_test.py
```
Validates the module loading, routing, retrieval, and output generation of the master pipeline.
