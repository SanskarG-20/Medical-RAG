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
        O --> L[Layout Agent<br>PyMuPDF + Tesseract OCR]
        O --> T[Table Agent]
        O --> I[Image Agent]
        O --> E[Entity Agent]
        O -.-> V{Validator Agent}
    end

    subgraph Multimodal RAG & Router
        V -- Validated Data --> DB[(ChromaDB / Neon Postgres)]
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

The **Layout Agent** uses a two-pass extraction strategy:
1. **Native text extraction** via PyMuPDF for digitally-born PDFs.
2. **OCR fallback** via Tesseract for scanned/image-based medical documents (renders each page at 300 DPI and runs OCR).

```mermaid
graph LR
    PDF[Raw Medical PDF] --> Parser[Parser Orchestrator]
    
    Parser --> Layout[Layout Agent<br>PyMuPDF + Tesseract OCR]
    Parser --> Table[Table Agent<br>Camelot/LLM]
    Parser --> Image[Image Agent<br>CLIP]
    Parser --> Entity[Entity Agent<br>SciSpacy]
    
    Layout --> Context[Context Aggregation]
    Table --> Context
    Image --> Context
    Entity --> Context
    
    Context --> Validator{Validator Agent}
    Validator -- Pass --> Structured[Structured Output<br>JSON/Dict]
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

Vector storage uses **ChromaDB** with persistent local storage, replacing FAISS for improved cross-platform compatibility (especially macOS Apple Silicon).

```mermaid
graph TD
    %% Hybrid Retrieval
    Router[Query & Routing Weights] --> HR[Hybrid Retriever]
    
    HR --> |Hard Facts| SQL[("SQL / PostgreSQL<br>Lab Values, Meds")]
    HR --> |Narrative| Vec[("ChromaDB<br>Clinical Notes, Papers")]
    HR --> |Visual| Multimodal[("ChromaDB Multimodal<br>X-rays, MRIs")]
    HR --> |Similar Cases| Graph[("Knowledge Graph")]

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
Make sure your environment is configured for Python 3.10+ and you have Tesseract OCR installed for scanned PDF support.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

### Install Dependencies
```bash
pip install -r requirements.txt
```

The SciSpacy biomedical model is installed automatically via the requirements. If you need to install it manually:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

### Note on Apple Silicon (M1/M2/M3) vs Windows/Linux

The vector store has been migrated from **FAISS** to **ChromaDB** to eliminate the known OpenMP / PyTorch threading lock conflicts (`[mutex.cc : 452] RAW: Lock blocking` / `zsh: segmentation fault`) on macOS Apple Silicon. ChromaDB uses a persistent local storage model that is fully cross-platform.

If you still encounter thread-related warnings, the `KMP_DUPLICATE_LIB_OK=True` workaround is pre-set in `app.py` and `run_test.py`. You can also manually export it:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Database Configuration
We use Neon PostgreSQL for scalable deployment. Be sure to configure your environment variables:
```bash
export DATABASE_URL="postgresql://user:password@neondb.aws.neon.tech/dbname"
```
*(If unset, it will fall back gracefully to local SQLite where supported)*

### API Keys
Configure the following in your `.env` file (see `.env.example`):
```
GROQ_API_KEY=your_groq_key_here       # Required for LLM reasoning (Llama 3 70B)
PINECONE_API_KEY=your_pinecone_key     # Optional
NCBI_API_KEY=your_ncbi_key            # Optional (PubMed retrieval)
```

### Running the Application

1. **Start the API Server:**
```bash
python app.py
```
This runs the FastAPI server with the orchestrated state machine on port 8000.

2. **Run the End-to-End Test:**
```bash
python run_my_test.py
```
Runs the full pipeline against a sample medical PDF: parse → chunk → embed into ChromaDB → retrieve → reason via Groq LLM.

3. **Run Module Tests:**
```bash
python run_test.py
```
Validates the module loading, routing, retrieval, and output generation of the master pipeline.

## Key Technology Choices

| Component | Technology | Why |
|---|---|---|
| PDF Text Extraction | PyMuPDF + Tesseract OCR | Handles both digital and scanned PDFs. Two-pass strategy: native text first, OCR fallback for image-based pages |
| Biomedical NER | SciSpacy (`en_core_sci_md`) | UMLS-grounded entity extraction for diseases, tests, chemicals |
| Vector Store | ChromaDB (persistent) | Cross-platform, no OpenMP conflicts, auto-persistent local storage |
| Multimodal Embeddings | CLIP (`openai/clip-vit-base-patch32`) | Joint text-image latent space for X-ray/MRI retrieval |
| Text Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) | Fast, normalized embeddings for clinical text similarity |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Filters low-relevance chunks to prevent hallucination |
| LLM Reasoning | Groq (Llama 3 70B) | Ultra-fast inference with strict JSON output + semantic verification |
| Database | Neon PostgreSQL / SQLite fallback | Scalable cloud DB with local development support |
| Query Routing | Dual-Signal (SciSpacy + MiniLM) | Entity grounding + semantic intent classification |
