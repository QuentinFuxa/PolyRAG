# 🧪 AI & RAG Service Toolkit

This repository provides a comprehensive toolkit for building and running advanced AI agent services using LangGraph, FastAPI, and Streamlit. It extends the original [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) by integrating sophisticated capabilities for **RAG (Retrieval-Augmented Generation) over both structured PostgreSQL databases and unstructured PDF documents**.

The toolkit features:
- **Advanced RAG Agent:** The primary agent, defined in `src/agents/pg_rag_assistant.py`, is capable of complex interactions involving database querying, document analysis, and visualization.
- **Database RAG:** Dynamically discovers PostgreSQL database schemas using `src/prompt_generator.py`, generates tailored prompts for the LLM, and allows the agent to query the database via SQL using the `execute_sql` tool.
- **Document RAG:** Processes PDFs using one of two backends (configurable via the `PDF_PARSER` environment variable):
    - **`nlm-ingestor` (Recommended):** Leverages `llmsherpa` to extract structured content and preserve the document's hierarchical structure (sections, paragraphs, lists). This allows the RAG system to understand context more effectively. Requires the [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) service to be running.
    - **`pymupdf`:** A faster, dependency-free alternative using the PyMuPDF library, but it does not inherently capture the document hierarchy.
- **Document Indexing:** Scripts `src/index-folder-script.py` (for local folders) and `src/index-urls-script.py` (for URLs from a DB table) handle the indexing of PDF content for vector and text search in PostgreSQL.
- **Interactive Visualization:** Features an advanced PDF viewer (`PDF_Viewer` tool) that highlights relevant sections found via RAG, and supports interactive Plotly graph generation (`Graph_Viewer` tool).
- **File Uploads:** The Streamlit interface currently supports uploading **PDF documents** directly through the chat for processing.
- **Core Agent Framework:** Built on LangGraph with support for tools, human-in-the-loop interruptions, and supervision.
- **Web Services:** A FastAPI backend serves the agent, and a Streamlit frontend provides a user-friendly chat interface.
- **Supporting Features:** Content moderation (LlamaGuard), feedback mechanism (LangSmith), Docker support, and testing.

This project offers a template for building and running your own agents using the LangGraph framework, demonstrating a complete setup from agent definition to user interface.

## Overview

### Application Screenshot

<img src="media/app_screenshot.png" width="600">

### Quickstart

Run directly in python:

```sh
# Set required environment variables (see .env.example)
# At least one LLM API key and DATABASE_URL are needed
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DATABASE_URL=postgresql://user:password@host:port/dbname' >> .env
# Optionally set PDF_PARSER (defaults to nlm-ingestor)
# echo 'PDF_PARSER=pymupdf' >> .env
# If using nlm-ingestor, ensure its service is running (e.g., via Docker)
# If using embeddings, ensure OPENAI_API_KEY is set

# uv is recommended but "pip install ." also works
pip install uv
uv sync --frozen
# "uv sync" creates .venv automatically
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

Run with docker:

```sh
# Set required environment variables in .env (as above)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DATABASE_URL=postgresql://user:password@host:port/dbname' >> .env
# Add LLMSHERPA_API_URL if nlm-ingestor runs on a different host/port for Docker
# echo 'LLMSHERPA_API_URL=http://host.docker.internal:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true' >> .env

docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Key Features

**1. Database RAG & Interaction (PostgreSQL):**
   - **Dynamic Schema Discovery:** `src/prompt_generator.py` automatically inspects specified database schemas, tables, columns, and data types.
   - **Intelligent Prompt Generation:** Creates tailored system prompts describing the discovered DB schema and available tools, guiding the LLM on how to query effectively. Includes optional LLM-generated column summaries and sample data.
   - **SQL Execution Tool:** Provides an `execute_sql` tool allowing the agent (`pg_rag_assistant.py`) to directly query the database.
   - **Special Column Identification:** Detects `vector` and `tsvector` columns, enabling guidance for semantic and full-text searches.

**2. Document RAG & Visualization (PDFs):**
   - **Flexible PDF Processing:** Choose between `nlm-ingestor` (preserves hierarchy) and `pymupdf` backends via the `PDF_PARSER` environment variable (see `src/rag_system.py`).
   - **Hierarchical Understanding (nlm-ingestor):** When using `nlm-ingestor`, the system captures the document's structure (levels, parent-child relationships), allowing for more context-aware retrieval.
   - **Vector & Text Search:** Indexes document chunks in PostgreSQL for efficient semantic (embedding, requires OpenAI API key) and keyword (full-text) search.
   - **Indexing:** Use `src/index-folder-script.py` for local PDFs or `src/index-urls-script.py` to index PDFs from URLs listed in a database table.
   - **Interactive PDF Viewer:** The `PDF_Viewer` tool displays source PDFs with relevant sections highlighted based on RAG results.
   - **Document Search Tools:** `Query_RAG` (keyword/semantic search) and `Query_RAG_From_ID` (retrieve specific blocks) enable interaction with indexed documents.

**3. Data Visualization:**
   - **Plotly Graph Generation:** The agent can generate and display interactive Plotly charts using the `Graph_Viewer` tool based on data analysis or user requests.

**4. Core Agent & Service Framework:**
   - **Primary Agent:** `src/agents/pg_rag_assistant.py` orchestrates database and document RAG.
   - **LangGraph Foundation:** Customizable agent architecture using LangGraph v0.3+ features.
   - **FastAPI Service:** Asynchronous backend serving the agent via REST API.
   - **Streamlit Interface:** User-friendly chat UI (`src/streamlit_app.py`) for interacting with the agent.

**5. Supporting Features:**
   - **PDF Upload:** Upload PDF files directly via the Streamlit chat interface for processing.
   - **Content Moderation:** Optional integration with LlamaGuard.
   - **Feedback Mechanism:** Star-based feedback system integrated with LangSmith tracing.
   - **Docker Support:** Comprehensive Dockerfiles and `compose.yaml`.
   - **Testing:** Unit and integration tests.

### Key Files

- `src/agents/pg_rag_assistant.py`: The main RAG agent definition.
- `src/agents/`: Other agent definitions and tool implementations (`tools_pg.py`, `tools_plotly.py`).
- `src/db_manager.py`: Manages PostgreSQL connections and schema discovery.
- `src/prompt_generator.py`: Generates dynamic system prompts for the agent based on DB schema and tools.
- `src/rag_system.py`: Handles PDF processing (choosing backend), indexing, and searching logic.
- `src/index-folder-script.py`: Script to index local PDF files.
- `src/index-urls-script.py`: Script to index PDF files from URLs in a database.
- `src/schema/`: Pydantic models for data structures and API communication.
- `src/core/`: Core modules like LLM configuration (`llm.py`) and settings (`settings.py`).
- `src/service/service.py`: The FastAPI application.
- `src/client/client.py`: Client library for interacting with the service.
- `src/streamlit_app.py`: The Streamlit chat interface.
- `tests/`: Unit and integration tests.
- `docker/`: Dockerfiles.
- `compose.yaml`: Docker Compose configuration.

## Setup and Usage

1.  Clone the repository:
    ```sh
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```
2.  Set up environment variables:
    Create a `.env` file in the root directory.
    - **LLM Configuration:** At least one LLM API key (e.g., `OPENAI_API_KEY`) is required.
    - **Database Connection:** Set `DATABASE_URL` to your PostgreSQL connection string (e.g., `DATABASE_URL=postgresql://user:password@host:port/dbname`). This is crucial for agent memory and RAG features.
    - **PDF Parser Backend (Optional):** Set `PDF_PARSER` to `nlm-ingestor` (default) or `pymupdf`.
    - **NLM Ingestor URL (Optional):** If using `nlm-ingestor` and it's running elsewhere (especially with Docker), set `LLMSHERPA_API_URL`.
    - **Embeddings (Optional):** Embeddings are used if `OPENAI_API_KEY` is set. The `RAGSystem` checks for this.
    - **Other Options:** See `src/core/settings.py` and the [`.env.example` file](./.env.example) for more options (other model providers, LangSmith, LlamaGuard, etc.).

3.  Run the agent service and Streamlit app using Python directly or Docker (see [Quickstart](#quickstart)). Docker is recommended for easier dependency management, especially for `nlm-ingestor`.

### Building or customizing your own agent

To customize the agent:

1.  Modify `src/agents/pg_rag_assistant.py` or add new agents in `src/agents/`.
2.  Import and add any new agents to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface (`src/streamlit_app.py`) if needed.

### Docker Setup

Use `docker compose watch` for development. See the [Quickstart](#quickstart) section for `.env` considerations, especially `DATABASE_URL` and potentially `LLMSHERPA_API_URL`.

- Access the Streamlit app at `http://localhost:8501`.
- The agent service API is at `http://0.0.0.0:8080` (OpenAPI docs at `/redoc`).
- Use `docker compose down` to stop.

### Building other apps on the AgentClient

Use `src/client/client.AgentClient` to interact with the service. See `src/run_client.py` for examples.

```python
from client import AgentClient
client = AgentClient() # Assumes service running locally

# Example: Ask the RAG agent a question
response = client.invoke(
    "What are the main risks mentioned in document INSSN-LYO-2023-0461?",
    agent="pg_rag_assistant" # Specify the agent
)
response.pretty_print()
```

### Development with LangGraph Studio

Install LangGraph Studio, add your `.env` file, and launch the studio pointed at the root directory. Customize `langgraph.json` as needed.

### Using Ollama

Experimental support. See the original README section if needed, ensuring environment variables like `OLLAMA_MODEL` and potentially `OLLAMA_BASE_URL` (for Docker) are set.

### Local development without Docker

See the [Quickstart](#quickstart) section for Python setup using `uv`.

## Contributing

Contributions are welcome! Please submit Pull Requests. Run tests locally:

```sh
# Ensure virtual environment is active
pip install uv
uv sync --frozen --dev # Install dev dependencies
pre-commit install
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note on PDF Processing Dependencies:**
- **`nlm-ingestor` backend:** Requires the [nlm-ingestor service](https://github.com/nlmatics/nlm-ingestor) to be running separately (e.g., via its Docker image). Set `LLMSHERPA_API_URL` in `.env` if it's not accessible at the default localhost URL expected by `src/rag_system.py`.
- **`pymupdf` backend:** Requires `pymupdf` Python package (`uv sync` handles this).
- **Embeddings:** Requires `openai` Python package and an `OPENAI_API_KEY` for semantic search capabilities.
