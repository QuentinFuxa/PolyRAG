import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

# Assuming DatabaseManager is accessible, adjust import if needed
# from .db_manager import DatabaseManager
# For now, using a placeholder import path
from db_manager import DatabaseManager

logger = logging.getLogger(__name__)

def generate_db_rag_prompt(
    db_name: str,
    schemas_to_include: List[str],
    tables_to_include: Optional[Dict[str, List[str]]] = None,
    assistant_name: str = "Database Query Assistant",
    db_type: str = "PostgreSQL",
    include_samples: bool = True,
    sample_size: int = 3
) -> str:
    """
    Generates a system prompt for an LLM to query a database based on discovered schema.

    Args:
        db_name: Name of the database (for informational purposes).
        schemas_to_include: List of schema names to inspect and include.
        tables_to_include: Optional dict mapping schema names to specific table lists.
                           If None, all tables in schemas_to_include are used.
        assistant_name: Name for the assistant in the prompt.
        db_type: Type of the database (e.g., PostgreSQL).
        include_samples: Whether to fetch and include column sample data.
        sample_size: Number of distinct samples to fetch per column.

    Returns:
        A formatted system prompt string.
    """
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        logger.error(f"Failed to initialize DatabaseManager: {e}")
        return f"Error: Could not connect to the database or initialize DatabaseManager. {e}"

    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt_lines = [
        f"You are a helpful {assistant_name}.",
        f"Today's date is {current_date}.",
        f"You are a helpful {assistant_name}.",
        f"Today's date is {current_date}.",
        f"You have access to a {db_type} database named '{db_name}' and tools to interact with associated documents.",
        "Below is a description of the relevant database tables/columns and document interaction tools.",
        "",
    ]

    table_descriptions = []
    prompt_lines.append("## Database Schema Information")

    for schema in schemas_to_include:
        try:
            all_tables_in_schema = db_manager.list_tables(schema)
            target_tables = all_tables_in_schema
            
            # Filter tables if specific list is provided for the schema
            if tables_to_include and schema in tables_to_include:
                target_tables = [tbl for tbl in all_tables_in_schema if tbl in tables_to_include[schema]]
                if not target_tables:
                    logger.warning(f"No specified tables found in schema '{schema}'. Skipping.")
                    continue
            elif not all_tables_in_schema:
                 logger.warning(f"No tables found in schema '{schema}'. Skipping.")
                 continue


            for table in target_tables:
                table_descriptions.append(f"### Table `{schema}`.`{table}`")
                table_descriptions.append("")
                table_descriptions.append("#### Columns")

                try:
                    columns = db_manager.get_table_columns(schema, table)
                    special_cols = db_manager.identify_special_columns(schema, table)
                    embedding_cols = special_cols.get('embedding', [])
                    tsvector_cols = special_cols.get('tsvector', [])

                    if not columns:
                        table_descriptions.append("  - (Could not retrieve column information)")
                        continue

                    for i, col in enumerate(columns):
                        col_name = col['name']
                        col_type = col['type']
                        table_descriptions.append(f"{i+1}. **Column Name**: `{col_name}`")
                        table_descriptions.append(f"   - **Type**: `{col_type}`")

                        # Add notes for special columns
                        if col_name in embedding_cols:
                            table_descriptions.append(f"   - **Note**: This column contains embeddings, suitable for semantic similarity searches (e.g., using vector operators like `<=>`).")
                        if col_name in tsvector_cols:
                             table_descriptions.append(f"   - **Note**: This column contains tsvector data, suitable for full-text keyword searches (e.g., using functions like `to_tsquery` and `@@`).")

                        if include_samples:
                            try:
                                samples = db_manager.get_column_samples(schema, table, col_name, n=sample_size)
                                if samples:
                                     # Format samples nicely, handling potential non-string types safely
                                    sample_str = ", ".join([repr(s) for s in samples])
                                    table_descriptions.append(f"   - **Example values**: `{sample_str}`")
                                else:
                                     table_descriptions.append(f"   - **Example values**: (No distinct non-null values found or sampled)")
                            except Exception as sample_error:
                                logger.error(f"Error sampling column {schema}.{table}.{col_name}: {sample_error}")
                                table_descriptions.append(f"   - **Example values**: (Error retrieving samples: {sample_error})")
                        table_descriptions.append("") # Add space between columns

                except Exception as table_error:
                    logger.error(f"Error processing table {schema}.{table}: {table_error}")
                    table_descriptions.append(f"  - (Error retrieving details for this table: {table_error})")
                
                table_descriptions.append("") # Add space between tables

        except Exception as schema_error:
            logger.error(f"Error processing schema {schema}: {schema_error}")
            prompt_lines.append(f"## Schema `{schema}`")
            prompt_lines.append(f"(Error retrieving tables for this schema: {schema_error})")
            prompt_lines.append("")

    prompt_lines.extend(table_descriptions)


    # --- Add RAG Tool Descriptions and Instructions ---
    prompt_lines.extend([
        "",
        "## Document Interaction Tools",
        "",
        "You also have access to tools for searching within and viewing associated documents (e.g., PDF reports referenced in the database).",
        "",
        "### Tool: Query_RAG",
        "- Use this tool to search for information within specific indexed documents (e.g., inspection reports).",
        "- Parameters:",
        "  - `query`: (List[str], required) List of keywords (not sentences) to search for. Example: [\"incendie\", \"risque\", \"confinement\"]",
        "  - `source_names`: (List[str], optional) Name(s) of the specific document(s) to search within. Example: [\"INSSN-LYO-2023-0461\"]. If omitted, searches across available documents.",
        "  - `get_children`: (bool, optional, default: true) Whether to include child blocks of matching results.",
        "  - `get_parents`: (bool, optional, default: false) Whether to include parent blocks of matching results.",
        "- Returns: Text blocks relevant to your search, including metadata like block IDs.",
        "",
        "### Tool: Query_RAG_From_ID",
        "- Use this tool to retrieve specific text blocks by their IDs, often used to navigate document structure based on initial `Query_RAG` results.",
        "- Parameters:",
        "  - `block_ids`: (Union[int, List[int]], required) A single block ID or a list of block IDs to retrieve.",
        "  - `source_name`: (str, optional) Name of the document the blocks belong to.",
        "  - `get_children`: (bool, optional, default: false) Whether to include child blocks.",
        "- Returns: The requested text blocks.",
        "",
        "### Tool: PDF_Viewer",
        "- Use this tool to display a specific PDF document with relevant sections highlighted.",
        "- Parameters:",
        "  - `pdf_file`: (str, required) Name of the PDF file (often derived from database queries or `Query_RAG` results, usually without the .pdf extension).",
        "  - `block_ids`: (List[int], required) List of block IDs (obtained from `Query_RAG` or `Query_RAG_From_ID`) to highlight in the PDF.",
        "- **IMPORTANT**: After using `Query_RAG` and/or `Query_RAG_From_ID` to find information in a document, ALWAYS call `PDF_Viewer` as the final step for that document interaction to display the highlighted context to the user. A button will appear for the user to view the PDF.",
        "",
        "### Tool: execute_sql", # Keep the SQL tool description
        "- Use this tool to execute SQL queries against the database described above.",
        "- Parameters:",
        f"  - `query`: (string, required) The SQL query to execute. Ensure it is valid for {db_type}.",
        "- Returns: The query results (list of records) or an error message.",
        "",
        "## General Instructions",
        "- Your primary goal is to answer user questions using the available database and document information.",
        "- **Determine the best approach:**",
        "  - If the question is about structured data, summaries, counts, or specific records identifiable via database columns -> Use the `execute_sql` tool with appropriate SQL queries based on the schema.",
        "  - If the question is about the *content* of specific documents (e.g., details within a report mentioned in the database) -> Use `Query_RAG` to search the document, potentially followed by `Query_RAG_From_ID` to navigate, and ALWAYS conclude with `PDF_Viewer` to show the highlighted document.",
        "  - You might need to use `execute_sql` first to find document identifiers (like names or links) before using `Query_RAG`.",
        "- When using `execute_sql`:",
        "  - Formulate queries based on the user's request and the database schema provided above.",
        "  - Analyze the results returned by the tool to formulate your answer.",
        f"  - Leverage special columns (embedding, tsvector) noted in the schema description for advanced searches if applicable (e.g., using vector operators like `<=>` or full-text functions like `to_tsquery` for {db_type}).",
        "- When using `Query_RAG`:",
        "  - Use specific keywords in the `query` parameter.",
        "  - Use document names (e.g., report names found via SQL) in the `source_names` parameter.",
        "- **Crucially**: After finding relevant information in a document using `Query_RAG`/`Query_RAG_From_ID`, ALWAYS call `PDF_Viewer` with the document name and the relevant `block_ids`.",
        "- Do not mention the database, SQL, or specific tool names (`execute_sql`, `Query_RAG`, etc.) explicitly in your final response to the user. Present the information naturally.",
        "- Summarize or group data where appropriate.",
        "- If a user command is `/debug` and mentions a document name, call `PDF_Viewer` with `debug=True` and the `pdf_file` parameter set to that name.", # Include debug instruction if needed
    ])

    return "\n".join(prompt_lines)

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Replace with your actual database name and desired schemas/tables
    test_db_name = "asnr_chat_db" # Example DB name from config
    # Schemas to inspect (e.g., 'public' or custom schemas)
    test_schemas = ['public'] 
    # Optional: Specify exact tables within schemas, or leave None to get all
    # test_tables = {'public': ['public_data', 'another_table']} 
    test_tables = None 
    # --- End Configuration ---

    print(f"Generating prompt for database '{test_db_name}', schemas: {test_schemas}...")
    
    # Ensure DB connection details are correctly set in config.yml or environment variables
    # for DatabaseManager to work.
    generated_prompt = generate_db_rag_prompt(
        db_name=test_db_name,
        schemas_to_include=test_schemas,
        tables_to_include=test_tables,
        include_samples=True,
        sample_size=3
    )

    print("\n--- Generated Prompt ---")
    print(generated_prompt)
    print("--- End Generated Prompt ---")

    # You might want to save this to a file:
    # with open("generated_prompt.txt", "w") as f:
    #     f.write(generated_prompt)
    # print("\nPrompt saved to generated_prompt.txt")
