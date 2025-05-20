# PolyRAG

A comprehensive toolkit for building and running advanced AI agent services with RAG capabilities over both **PostgreSQL databases** and **PDF documents**.

## Overview

PolyRAG extends the original [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) by integrating sophisticated capabilities for Retrieval-Augmented Generation across structured and unstructured data sources. Built on LangGraph, FastAPI, and Streamlit, it provides a complete framework from agent definition to user interface.

![Application Screenshot](media/demo_1.png)
![Application Screenshot](media/demo_2.png)
![Application Screenshot](media/demo_3.png)

## Features

- **Advanced RAG Agent:** Handles complex interactions involving database querying, document analysis, and visualization

![Example](media/example.png)

- **Core Agent Tools:** The `pg_rag_assistant` is equipped with a suite of powerful tools:
  | Tool Name           | Description                                                                                                                               |
  |---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
  | `execute_sql`       | Enables secure, read-only SQL execution for dynamic data retrieval. |
  | `tool_graphing_agent` | Invokes a specialized agent to create visualizations from natural language and data (direct input or SQL query). |
  | `query_rag`         | Leverages Retrieval Augmented Generation to intelligently search documents (specified by name or SQL) for keywords, extracting precise text blocks and context. |
  | `query_rag_from_id` | Provides targeted access to document content by retrieving specific blocks via their indices, including optional child and surrounding blocks. |
  | `highlight_pdf`     | Prepares PDF documents for frontend display with highlighted content blocks. |

- **Database RAG:**
  - Dynamically discovers PostgreSQL database schemas
  - Generates tailored prompts for the LLM
  - Enables secure SQL querying via the `execute_sql` tool
  
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

## Configuration

Create a `.env` file with the following options:

- `OPENAI_API_KEY`: Required for LLM and embeddings
- `DATABASE_URL`: PostgreSQL connection string
- `SCHEMA_APP_DATA`: Database schema for application data. Default: `document_data`
- `PDF_PARSER`: `nlm-ingestor` (default) or `pymupdf`
- `UPLOADED_PDF_PARSER`: Parser used for the chat uploaded documents. `nlm-ingestor` (default) or `pymupdf`
- `LLMSHERPA_API_URL`: URL for nlm-ingestor service if `nlm-ingestor` is used
- `SYSTEM_PROMPT_PATH`: Path to a system prompt file that you can generate using `scripts/prompt_generator.py`
- `TS_QUERY_LANGUAGE`: Language used for text search queries. Default: `english`

- Additional options in `src/core/settings.py`

## Key Files

- `src/agents/pg_rag_assistant.py`: Main RAG agent definition
- `src/db_manager.py`: PostgreSQL connection and schema discovery
- `scripts/prompt_generator.py`: Dynamic system prompt generation
- `src/rag_system.py`: PDF processing and search logic
- `scripts/index-folder-script.py`: Script to index local PDFs
- `scripts/index-urls-script.py`: Script to index PDFs from URLs
- `src/streamlit_app.py`: Chat interface
- `docker/`: Dockerfiles and `compose.yaml`

## Utility Scripts

These scripts help with indexing data and generating prompts. Ensure your environment is activated (`source .venv/bin/activate`) and necessary environment variables (like `DATABASE_URL`, `OPENAI_API_KEY`) are set in your `.env` file before running them.

### Indexing PDFs from a Folder (`scripts/index-folder-script.py`)

Indexes PDF documents from a local directory into the database and RAG system.

```sh
python scripts/index-folder-script.py --dir /path/to/your/pdf/folder [--embeddings]
```

- `--dir`: (Required) Path to the directory containing PDF files.
- `--pdf`: (Optional) Path to a single PDF file to index instead of a directory.
- `--embeddings`: (Optional) Generate and store embeddings for the documents (requires `OPENAI_API_KEY`).

### Indexing PDFs from URLs (`scripts/index-urls-script.py`)

Fetches PDF URLs from a specified database table, downloads them, and indexes their content.

```sh
python scripts/index-urls-script.py [--schema SCHEMA_NAME] [--table TABLE_NAME] [--column COLUMN_NAME] [--embeddings]
```

- `--schema`: (Optional) Database schema containing the table with URLs (default: `public`).
- `--table`: (Optional) Database table name containing the URLs (default: `arxiv_qbio_metadata_2025`).
- `--column`: (Optional) Column name within the table that holds the PDF URLs (default: `pdf_url`).
- `--embeddings`: (Optional) Generate and store embeddings for the documents (requires `OPENAI_API_KEY`).

### Generating the Database RAG Prompt (`scripts/prompt_generator.py`)

Inspects the database schema and generates a detailed system prompt for the RAG agent, saving it to `system_prompt.txt`.

```sh
python scripts/prompt_generator.py
```

This script reads database connection details from the environment and currently inspects the `public` schema by default. You can modify the script (`if __name__ == '__main__':` block) to target different schemas or tables if needed.

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
    "What are the main risks mentioned in document 2503_01910 v1?",
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


uv pip install langchain_mistralai
