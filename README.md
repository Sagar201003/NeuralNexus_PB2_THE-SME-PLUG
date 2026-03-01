# 🧬 SME-PLUG: Universal Subject Matter Expert Plugin

> A hot-swappable domain expertise plugin that injects specialized knowledge, structured decision trees, and source-of-truth citations into any AI agent.

---

## 🏗️ Architecture

```
Query → Domain Router (3-Layer Cascade) → Capsule Loader → Advanced RAG Pipeline → Expert LLM → Guardrails → Response

Domain Detection:
  L1: Keyword/Regex (< 5ms)
  L2: Zero-shot LLM Classifier (< 500ms)
  L3: Embedding Cosine Similarity (< 200ms)

Advanced RAG:
  HyDE → Hybrid BM25+Dense Retrieval → RRF Fusion → Cross-Encoder Reranking
```

## 📁 Project Structure

```
SME-PLUG/
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
│
├── core/                      # Core engine
│   ├── expert_core.py         # Main orchestrator
│   ├── capsule_loader.py      # YAML capsule loader
│   ├── domain_router.py       # 3-layer domain detection
│   └── confidence_gate.py     # Confidence thresholding
│
├── rag/                       # Advanced RAG pipeline
│   ├── vector_store.py        # ChromaDB per-capsule collections
│   ├── ingestion.py           # Document ingestion + BM25 indexing
│   ├── hyde_engine.py          # HyDE query expansion
│   ├── advanced_retriever.py  # Hybrid BM25+Dense+RRF
│   └── reranker.py            # Cross-encoder reranker
│
├── capsules/                  # DNA Capsules (domain expertise bundles)
│   ├── structural_engineering/
│   ├── cybersecurity/
│   └── legal/
│
├── guardrails/                # Response safety
│   ├── hallucination_detector.py
│   ├── citation_enforcer.py
│   └── output_validator.py
│
├── adapters/                  # Framework integrations
│   ├── langchain_adapter.py
│   └── crewai_adapter.py
│
├── api/                       # FastAPI server
│   ├── server.py
│   ├── routes.py
│   └── models.py
│
├── cli/                       # CLI tools
│   └── capsule_creator.py
│
└── demo/                      # Demo scripts
    ├── run_demo.py
    └── demo_queries.py
```

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

### 3. Run Demo
```bash
python main.py demo
```

### 4. Start API Server
```bash
python main.py api
# Server at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. One-Shot Query
```bash
python main.py query "Is this beam safe for 500 kN load?"
python main.py query "Triage this CVE-2024-3400 alert" --domain cybersecurity
```

## 🧬 Creating Custom Capsules

```bash
python main.py capsule create --domain "Petroleum Engineering" --docs ./pdfs/
python main.py ingest --domain petroleum_engineering
```

## 🔌 Framework Integration

### LangChain
```python
from core.expert_core import ExpertCore
from adapters.langchain_adapter import LangChainAdapter

ec = ExpertCore()
adapter = LangChainAdapter(ec)

# As a Tool
tool = adapter.create_tool()
agent.tools.append(tool)

# As a Retriever
retriever = adapter.create_retriever()
```

### CrewAI
```python
from adapters.crewai_adapter import CrewAIAdapter

adapter = CrewAIAdapter(ec)
expert_agent = adapter.create_expert_agent()
```

## 🛡️ Built-in Domains

| Domain | Expert | Standards |
|--------|--------|-----------|
| 🏗️ Structural Engineering | Senior Structural Engineer | AISC 360, IS 456, IS 800 |
| 🛡️ Cybersecurity | SOC Analyst | MITRE ATT&CK, NIST 800-53 |
| ⚖️ Legal | Contract Lawyer | UCC, Restatement of Contracts |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query with auto domain detection |
| GET | `/capsules` | List loaded capsules |
| POST | `/capsule/ingest` | Ingest docs for a capsule |
| GET | `/health` | Health check |

---
<img width="1307" height="722" alt="image" src="https://github.com/user-attachments/assets/e287b8fb-fe8e-4e45-8705-72b6cd7f61a5" />

**Built for the SME-PLUGIN Hackathon Challenge** 🏆
