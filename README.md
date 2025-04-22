# 🧪 AI & RAG Service Toolkit

This repository provides a comprehensive toolkit for building and running AI agent services using LangGraph, FastAPI, and Streamlit. It extends the original [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) by integrating advanced capabilities for **RAG (Retrieval-Augmented Generation) over both structured databases and unstructured PDF documents**.

The toolkit includes:
- **Database RAG:** Dynamically discovers PostgreSQL database schemas, generates tailored prompts for the LLM, and allows the agent to query the database via SQL.
- **Document RAG:** Processes PDFs using llmsherpa, indexes content for vector and text search in PostgreSQL, and enables searching within documents.
- **Interactive Visualization:** Features an advanced PDF viewer that highlights relevant sections found via RAG, and supports Plotly graph generation.
- **Core Agent Framework:** Built on LangGraph with support for tools, human-in-the-loop interruptions, and supervision.
- **Web Services:** A FastAPI backend serves the agent, and a Streamlit frontend provides a user-friendly chat interface.
- **Supporting Features:** File uploads, content moderation (LlamaGuard), feedback mechanism (LangSmith), Docker support, and testing.

It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

This project offers a template for you to easily build and run your own agents using the LangGraph framework. It demonstrates a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects by providing a full, robust toolkit.

## Overview

### [Try the app!](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Quickstart

Run directly in python

```sh
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

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

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Key Features

**1. Database RAG & Interaction (PostgreSQL):**
   - **Dynamic Schema Discovery:** Automatically inspects specified database schemas, tables, columns, and data types.
   - **Intelligent Prompt Generation:** Creates tailored system prompts describing the discovered DB schema, guiding the LLM on how to query effectively.
   - **SQL Execution Tool:** Provides an `execute_sql` tool allowing the agent to directly query the database.
   - **Sample Data Fetching:** Optionally includes example values from columns in the prompt for better context.
   - **Special Column Identification:** Detects `vector` and `tsvector` columns, enabling guidance for semantic and full-text searches.

**2. Document RAG & Visualization (PDFs):**
   - **PDF Processing (llmsherpa):** Extracts structured content, text, and layout information from PDF documents.
   - **Vector & Text Search:** Indexes document chunks in PostgreSQL for efficient semantic (embedding) and keyword (full-text) search.
   - **Interactive PDF Viewer:** Displays source PDFs with relevant sections highlighted based on RAG results, using tools like `Query_RAG`, `Query_RAG_From_ID`, and `PDF_Viewer`.
   - **Dependency:** Requires the [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) service for PDF processing (see setup note below).

**3. Data Visualization:**
   - **Plotly Graph Generation:** The agent can generate and display interactive Plotly charts based on data analysis or user requests.

**4. Core Agent & Service Framework:**
   - **LangGraph Agent:** Customizable agent architecture using LangGraph v0.3+ features (interruptions, commands, supervision).
   - **FastAPI Service:** Asynchronous backend serving the agent via REST API with streaming support.
   - **Streamlit Interface:** User-friendly chat UI for interacting with the agent.
   - **Multiple Agent Support:** Easily extendable to host multiple distinct agents.

**5. Supporting Features:**
   - **File Upload & Processing:** Allows users to upload files for the agent to potentially process or index.
   - **Content Moderation:** Optional integration with LlamaGuard (requires Groq API key).
   - **Feedback Mechanism:** Star-based feedback system integrated with LangSmith tracing.
   - **Docker Support:** Comprehensive Dockerfiles and `compose.yaml` for development and deployment.
   - **Testing:** Unit and integration tests covering various components.

### Key Files

- `src/agents/`: Agent definitions (e.g., combining tools and logic). Includes `tools.py` where tools like `execute_sql` are defined.
- `src/db_manager.py`: Manages PostgreSQL connections and includes schema discovery logic for Database RAG.
- `src/prompt_generator.py`: Contains the function to generate dynamic system prompts for the agent based on discovered DB schema and available tools.
- `src/rag_system.py`: Handles PDF processing (via llmsherpa), indexing document chunks, and searching logic for Document RAG.
- `src/schema/`: Defines Pydantic models for data structures and API communication.
- `src/core/`: Core modules like LLM configuration and application settings.
- `src/service/service.py`: The FastAPI application serving the agent(s).
- `src/client/client.py`: A client library for interacting with the agent service.
- `src/streamlit_app.py`: The Streamlit chat interface application.
- `tests/`: Unit and integration tests.
- `docker/`: Dockerfiles for the services.
- `compose.yaml`: Docker Compose configuration.

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/JoshuaC215/agent-service-toolkit.git
   cd agent-service-toolkit
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory.
   - **LLM Configuration:** At least one LLM API key (e.g., `OPENAI_API_KEY`) is required.
   - **Database Connection:** Set the `DATABASE_URL` variable to your PostgreSQL connection string (e.g., `DATABASE_URL=postgresql://user:password@host:port/dbname`). This is crucial for both agent memory and the RAG features.
   - **Other Options:** See the [`.env.example` file](./.env.example) for a full list of available environment variables (other model providers, LangSmith tracing, LlamaGuard, etc.).

3. You can now run the agent service and the Streamlit app locally, either with Docker or just using Python. The Docker setup is recommended for simpler environment setup and immediate reloading of the services when you make changes to your code.

### Building or customizing your own agent

To customize the agent for your own use case:

1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines two services: `agent_service` and `streamlit_app`. The `Dockerfile` for each is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose (>=[2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

3. The services will now automatically update when you make changes to your code:
   - Changes in the relevant python files and directories will trigger updates for the relevantservices.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

4. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

5. The agent service API will be available at `http://0.0.0.0:8080`. You can also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

6. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

### Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
# ================================== Ai Message ==================================
#
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

```

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio), a new IDE for developing agents in LangGraph.

You can simply install LangGraph Studio, add your `.env` file to the root directory as described above, and then launch LangGraph studio pointed at the root directory. Customize `langgraph.json` as needed.

### Using Ollama

⚠️ _**Note:** Ollama support in agent-service-toolkit is experimental and may not work as expected. The instructions below have been tested using Docker Desktop on a MacBook Pro. Please file an issue for any challenges you encounter._

You can also use [Ollama](https://ollama.com) to run the LLM powering the agent service.

1. Install Ollama using instructions from https://github.com/ollama/ollama
1. Install any model you want to use, e.g. `ollama pull llama3.2` and set the `OLLAMA_MODEL` environment variable to the model you want to use, e.g. `OLLAMA_MODEL=llama3.2`

If you are running the service locally (e.g. `python src/run_service.py`), you should be all set!

If you are running the service in Docker, you will also need to:

1. [Configure the Ollama server as described here](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server), e.g. by running `launchctl setenv OLLAMA_HOST "0.0.0.0"` on MacOS and restart Ollama.
1. Set the `OLLAMA_BASE_URL` environment variable to the base URL of the Ollama server, e.g. `OLLAMA_BASE_URL=http://host.docker.internal:11434`
1. Alternatively, you can run `ollama/ollama` image in Docker and use a similar configuration (however it may be slower in some cases).

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   pip install uv
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects built with or inspired by agent-service-toolkit

The following are a few of the public projects that drew code or inspiration from this repo.

- **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** A Next.JS frontend for agent-service-toolkit
- **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request editing the README or open a discussion with any new ones to be added!** Would love to include more projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Currently the tests need to be run using the local development without Docker setup. To run the tests for the agent service:

1. Ensure you're in the project root directory and have activated your virtual environment.

2. Install the development dependencies and pre-commit hooks:

   ```sh
   pip install uv
   uv sync --frozen
   pre-commit install
   ```

3. Run the tests using pytest:

   ```sh
   pytest
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note on PDF Processing Dependency:**
The Document RAG features rely on processing PDFs via the [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) service, which uses llmsherpa. You will need to run this service separately (e.g., using its Docker image) and potentially configure its URL in the environment variables (`LLMSHERPA_API_URL`) if it's not running on the default localhost address expected by `src/rag_system.py`.
