<p align="center">
  <img src="media/polyrag.svg" alt="PolyRAG Logo" width="200"/>
</p>

**Agentic RAG for Small LLMs — Modular, Orchestrated, and Context-Smart**

---

PolyRAG is an **agentic Retrieval-Augmented Generation (RAG) framework** designed to empower **small and local LLMs**: At its core, PolyRAG is built around modular, orchestrated agents—each specialized for a class of tasks and able to coordinate powerful toolchains. Every aspect is optimized for limited context windows, slower models, and on-premise or privacy-focused deployments. 

---

## Agentic Architecture

- **Agents** are modular, specialized entities that coordinate complex tasks, delegate subtasks to tools, and manage context flow.
- **Agent Orchestration:** Agents can call other agents or tools, pipe outputs directly, and adaptively route information to minimize context usage.
- **Agent-Tool Synergy:** Tools are robust to imperfect inputs and return concise, high-quality outputs. Agents decide when to use which tool, and how to chain them for optimal results.
- **Direct-to-User Output:** Tools and agents can send data directly to the user, bypassing the main agent to preserve context and maximize efficiency.

This agentic design is what enables PolyRAG to get the most out of small or local LLMs, supporting advanced workflows that would otherwise overwhelm limited context windows.


### Conversational Research Assistant (Agent-Orchestrated)
<img src="media/demo_1.png" alt="PolyRAG Chat Interface" width="800"/>

*Ask complex research questions and get precise, sourced answers. Accesses lexicon, finds paragraphs in documents*

---

### Suggestion buttons for the user
<img src="media/demo_3.png" alt="PolyRAG Smart Actions" width="800"/>

*Choose from a menu of advanced actions: synthesize literature, run SQL, generate graphs, and more. Each action is handled by specialized agents and tools, minimizing context usage.*

---

### Contextual PDF Highlighting
<img src="media/demo_4.png" alt="PolyRAG PDF Highlighting" width="800"/>

*View PDFs with automatically highlighted, contextually relevant blocks—extracted.*

---

### Data Visualization from Natural Language (Agent-Tool Chaining)
<img src="media/demo_2.png" alt="PolyRAG Data Visualization" width="800"/>

*Generate publication trend graphs and other visualizations from natural language requests, with results piped through the agent-tool chain.*

---

### End-to-End Agent Orchestration
<img src="media/example.png" alt="PolyRAG Agent Orchestration" width="800"/>

*See how PolyRAG chains SQL, RAG, and PDF tools to answer technical questions—each step coordinated by agents for context efficiency.*

---

### Architecture: Built for Agentic Workflows
<img src="media/schema-data.png" alt="PolyRAG Architecture" width="800"/>

*Agents and tools are designed to pipe outputs directly, auto-correct imperfect inputs, and minimize main agent context load. Every feature is built for small, slow, or local LLMs.*

---

## Document Extraction & Indexing

- **Semi-structured Extraction:**  
  - Uses NLM Ingestor and Tika with data type detection and tree structure.
  - Localization and regex rules for extracting structured parts.
  - Produces a structure with type, parent, child, and position.

- **Indexing:**  
  - Uses PostgreSQL TSVector (French) for efficient, scalable full-text search with tokenization and stemming.
  - No embeddings by default: lighter, scalable, and future-ready for on-premise models.
  - Excellent performance for technical queries.

---

## Features Overview

- Agentic RAG: Modular agents for database and document queries.
- Interactive PDF viewer with contextual highlights.
- Natural language to SQL and graphing, coordinated by agents.
- Conversation history, feedback, and moderation.
- Docker and Python support for easy deployment.


## ⚡ Quick Start

### Run with Python

```sh
# Set required environment variables
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DATABASE_URL=postgresql://user:password@host:port/dbname' >> .env

# Install dependencies (uv recommended)
pip install uv
uv sync --frozen
source .venv/bin/activate

# Run the service
python src/run_service.py

# In another terminal
source .venv/bin/activate
streamlit run src/streamlit-app.py
```

### Run with Docker

```sh
# Set required environment variables in .env
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DATABASE_URL=postgresql://user:password@host:port/dbname' >> .env

# Run with Docker Compose
docker compose watch
```

---

## ⚙️ Configuration

Create a `.env` file with the following options:

- `OPENAI_API_KEY`: Required for LLM and embeddings
- `DATABASE_URL`: PostgreSQL connection string
- `SCHEMA_APP_DATA`: Database schema for application data. Default: `document_data`
- `PDF_PARSER`: `nlm-ingestor` (default) or `pymupdf`
- `UPLOADED_PDF_PARSER`: Parser used for the chat uploaded documents. `nlm-ingestor` (default) or `pymupdf`
- `LLMSHERPA_API_URL`: URL for nlm-ingestor service if `nlm-ingestor` is used
- `SYSTEM_PROMPT_PATH`: Path to a system prompt file that you can generate using `scripts/prompt_generator.py`
- `LANGUAGE`: Language used for text search queries. Default: `english`

- Additional options in `src/core/settings.py`

