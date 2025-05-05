# src/display_texts.py

# App constants
APP_TITLE = "PolyRAG"
APP_ICON = ":material/experiment:"
AI_ICON = ":material/flare:"
USER_ICON = ":material/person:"

# PDF Dialog
PDF_DIALOG_TITLE = "Document"
PDF_DIALOG_DEBUG_PREFIX = "/debug"
PDF_DIALOG_NO_PDF_ERROR = "No PDF selected for viewing."
PDF_DIALOG_CLOSE_BUTTON = "Close"

# Agent Connection
AGENT_CONNECTING_SPINNER = "Connecting to agent service..."
AGENT_CONNECTION_ERROR = "Error connecting to agent service at {agent_url}: {e}"
AGENT_CONNECTION_RETRY_MSG = "The service might be booting up. Try again in a few seconds."

# Conversation Loading
CONVERSATION_LOADING_SPINNER = "Loading conversation..."
CONVERSATION_LOAD_ERROR = "Error loading conversation: {e}"
CONVERSATION_NOT_FOUND_ERROR = "No message history found for this Thread ID."
DEFAULT_CONVERSATION_TITLE = "New conversation"

# Sidebar
SIDEBAR_HEADER = "Platform for database and PDF document RAG, with visualization capabilities.\nPowered by LangGraph, PostgreSQL, Plotly and PDF processors. Built with FastAPI and Streamlit."
NEW_CONVERSATION_BUTTON = "**New conversation**"
EDIT_TITLE_INPUT_LABEL = "Conversation title"
SAVE_TITLE_BUTTON = "Save"
CANCEL_TITLE_BUTTON = "Cancel"
RECENT_SUBHEADER = "Recent"
CONVERSATION_LAST_UPDATED_HELP = "Last updated: {date_str}"
EDIT_CONVERSATION_TITLE_HELP = "Edit conversation title"
DELETE_CONVERSATION_HELP = "Delete this conversation"
LOAD_CONVERSATIONS_ERROR = "Error loading conversations: {e}"
SETTINGS_POPOVER_LABEL = ":material/settings: Settings"
SELECT_LLM_LABEL = "LLM to use"
SELECT_AGENT_LABEL = "Agent to use"
STREAMING_TOGGLE_LABEL = "Stream results"

# Welcome Messages
WELCOME_CHATBOT = "Hello! I'm a simple chatbot. Ask me anything!"
WELCOME_INTERRUPT = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
WELCOME_RESEARCH = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
WELCOME_PG_RAG = "Hello! I'm an AI-powered research assistant with Postgres database access, PDF analysis, and visualization tools. I can search documents, run SQL queries, and save our conversations for later reference. Ask me anything!"
WELCOME_DEFAULT = "Hello! I'm an AI agent. Ask me anything!"

# Example Prompts
PROMPT_BUTTON_DB_QUERY = "How many preprints in the q-bio.PE category since the beginning of the year?"
PROMPT_SUGGESTED_DB_QUERY = "How many preprints in the q-bio.PE category since the beginning of the year?"
PROMPT_BUTTON_DEBUG_PDF = "/debug 2504.12888v1"
PROMPT_SUGGESTED_DEBUG_PDF = "/debug 2504.12888v1"
PROMPT_BUTTON_CREATE_GRAPH = "Plot a graph of anemia distribution by condition and age for the year 2012"
PROMPT_SUGGESTED_CREATE_GRAPH = "Plot a graph of anemia distribution by condition and age for the year 2012"
PROMPT_BUTTON_DOCUMENT_SUMMARY = "Summarize the most recent preprint"
PROMPT_SUGGESTED_DOCUMENT_SUMMARY = "Summarize the most recent preprint"

# Chat Input & File Upload
CHAT_INPUT_PLACEHOLDER = "Your message"
FILE_ATTACHED_BADGE = ":violet-badge[:material/description: {file_name}] "
FILE_UPLOADING_STATUS = "File being uploaded..."
FILE_UPLOAD_SUCCESS_STATUS = "File {file_name} uploaded successfully!"
FILE_UPLOAD_ERROR_STATUS = "Error uploading {file_name}: {e}"
FILES_UPLOADED_COMPLETE_STATUS = "{count} files uploaded"

# Title Generation
TITLE_GENERATION_PROMPT = "Generate a short title (< 50 chars) summarizing this conversation. First user message: {user_text}"
RESPONSE_GENERATION_ERROR = "Error generating response: {e}"

# Message Drawing & Tool Calls
UNEXPECTED_MESSAGE_TYPE_ERROR = "Unexpected message type: {msg_type}"
TOOL_CALL_STATUS = "Tool Call: {tool_name}"
TOOL_CALL_INPUT_LABEL = "Input:"
TOOL_CALL_OUTPUT_LABEL = "Output :"
UNEXPECTED_CHATMESSAGE_TYPE_ERROR = "Unexpected ChatMessage type: {msg_type}"
GRAPH_RETRIEVAL_STATUS = "Retrieving graph with ID: {graph_id}"
GRAPH_RETRIEVED_SUCCESSFULLY = "Graph retrieved successfully"
GRAPH_NON_JSON_DATA = "Retrieved non-JSON graph data"
GRAPH_NO_DATA_RETURNED = "No graph data returned"
GRAPH_RETRIEVAL_ERROR = "Error retrieving graph: {e}"
VIEW_PDF_BUTTON = "View PDF: {pdf_name}"
PDF_READY_STATUS = "PDF ready: {pdf_name}"
PDF_PROCESSING_ERROR = "Error processing PDF: {e}"
RAW_OUTPUT_LABEL = "Raw output: {content}"
UNEXPECTED_CUSTOMDATA_ERROR = "Unexpected CustomData message received from agent"

# Feedback
FEEDBACK_STARS_KEY = "stars"
FEEDBACK_HUMAN_INLINE_COMMENT = "In-line human feedback"
FEEDBACK_SAVE_ERROR = "Error recording feedback: {e}"
FEEDBACK_SAVED_TOAST = "Feedback recorded"
FEEDBACK_TEXT_AREA_LABEL = "Additional feedback (optional)"
FEEDBACK_TEXT_AREA_PLACEHOLDER = "Please provide any additional comments or suggestions..."
FEEDBACK_SUBMIT_BUTTON = "Submit Feedback"
FEEDBACK_SUBMITTED_TOAST = "Detailed feedback submitted. Thank you!"
FEEDBACK_STARS_ICON = ":material/reviews:"
