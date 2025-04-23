# PolyRAG

A comprehensive toolkit for building and running advanced AI agent services with RAG capabilities over both **PostgreSQL databases** and **PDF documents**.

## Overview

PolyRAG extends the original [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) by integrating sophisticated capabilities for Retrieval-Augmented Generation across structured and unstructured data sources. Built on LangGraph, FastAPI, and Streamlit, it provides a complete framework from agent definition to user interface.

![Application Screenshot](media/app_screenshot.png)

## Features

- **Advanced RAG Agent:** Handles complex interactions involving database querying, document analysis, and visualization
- **Database RAG:** 
  - Dynamically discovers PostgreSQL database schemas
  - Generates tailored prompts for the LLM
  - Enables SQL querying via the `execute_sql` tool
  
- **Document RAG:** Two configurable backends:
  - **`nlm-ingestor` (Recommended):** Preserves document hierarchy for better context understanding
  - **`pymupdf`:** Faster alternative with simpler text extraction
  
- **Document Processing:**
  - Indexing scripts for local folders and URLs
  - Vector and text search in PostgreSQL
  - Interactive PDF viewer with relevant section highlighting
  
- **Visualization:** Interactive Plotly graph generation
- **File Uploads:** Upload PDFs directly through the chat interface
- **Conversation History:** Stored in PostgreSQL with auto-generated titles, accessible via sidebar
- **Supporting Features:** Content moderation, feedback mechanism, Docker support, and testing

## Quick Start

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
streamlit run src/streamlit_app.py
```

### Run with Docker

```sh
# Set required environment variables in .env
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DATABASE_URL=postgresql://user:password@host:port/dbname' >> .env

# Run with Docker Compose
docker compose watch
```

## Architecture

![Agent Architecture](media/agent_architecture.png)

## Configuration

Create a `.env` file with the following options:

- `OPENAI_API_KEY`: Required for LLM and embeddings
- `DATABASE_URL`: PostgreSQL connection string
- `PDF_PARSER`: `nlm-ingestor` (default) or `pymupdf`
- `LLMSHERPA_API_URL`: URL for nlm-ingestor service if not using default
- Additional options in `src/core/settings.py`

## Key Files

- `src/agents/pg_rag_assistant.py`: Main RAG agent definition
- `src/db_manager.py`: PostgreSQL connection and schema discovery
- `src/prompt_generator.py`: Dynamic system prompt generation
- `src/rag_system.py`: PDF processing and search logic
- `src/index-folder-script.py`: Script to index local PDFs
- `src/index-urls-script.py`: Script to index PDFs from URLs
- `src/streamlit_app.py`: Chat interface
- `docker/`: Dockerfiles and `compose.yaml`

## Customization

To build your own agent:

1. Modify `src/agents/pg_rag_assistant.py` or add new agents
2. Add agents to the `agents` dictionary in `src/agents/agents.py`
3. Adjust the Streamlit interface as needed

## Client Usage

```python
from client import AgentClient
client = AgentClient() # Assumes service running locally

# Ask the RAG agent a question
response = client.invoke(
    "What are the main risks mentioned in document INSSN-LYO-2023-0461?",
    agent="pg_rag_assistant"
)
response.pretty_print()
```

## Development

```sh
# Install dev dependencies
pip install uv
uv sync --frozen --dev
pre-commit install
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
