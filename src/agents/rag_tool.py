import json
from typing import Dict, List, Optional, Any, Union, Tuple, Union
from enum import Enum
import os
import psycopg2 # Import psycopg2 for error handling
from langchain_core.tools import BaseTool, tool

from rag_system import RAGSystem, SearchStrategy
from db_manager import schema_app_data # Import schema_app_data

# Initialize the RAG system
# Check if embeddings should be used based on environment variable
use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() in ("true", "1", "yes")
rag_system = RAGSystem(use_embeddings=use_embeddings)

class SearchStrategyEnum(str, Enum):
    TEXT = "text"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


def query_rag_func(
    keywords: List[str], # Words to search for in the documents. Example: ["isotope & atoms", "molecule", "reaction"]. Use "&" to require to have both words (or more) in the result.
    source_query: Optional[str] = None, # SQL query string that returns a single column containing document names. Example: "SELECT doc_name FROM documents WHERE year = 2025"
    source_names: Optional[List[str]] = None, # List of document names to search within. Example: ["report_campaign", "biology_comparison_report"]
    get_children: bool = True, # Whether to retrieve child blocks for each found block.
    max_results_per_source: Optional[int] = None # Maximum number of results to return per source document. Defaults to 3 in the RAG system.
    ) -> str:
    """
    Query the RAG system to find relevant information in documents. Provide EITHER source_names OR source_query.

    Args:
        keywords: Words to search for in the documents. Example: ["isotope & atoms", "molecule", "reaction"]. Use "&" to require to have both words (or more) in the result.
        source_query: A SQL query string that returns a single column containing document names. Example: "SELECT doc_name FROM documents WHERE year = 2025". The query MUST start with SELECT and return only one column. Use this OR source_names.
        source_names: A list of document names to search within. Example: ["report_campaign", "biology_comparison_report"]. Use this OR source_query.
        get_children: Whether to retrieve child blocks for each found block. Defaults to True.
        max_results_per_source: Maximum number of results to return per source document. If None, defaults to 3.

    Returns:
        The structured result from the RAG system, typically a dictionary containing 'success' status and 'results' list (grouped by document), or an error dictionary.
            - document name
            - id of the block
            - text content of the block
            - page number of the block
            - parent block id
            - level of the block
            - tag of the block (header, list_item, etc.)
        Or an error dictionary if something goes wrong.
    """
    final_source_names = []

    # Validate that exactly one of source_names or source_query is provided
    if (source_names is None and source_query is None) or \
       (source_names is not None and source_query is not None):
        return {"error": "Provide either 'source_names' (list of strings) or 'source_query' (SQL string), but not both."}

    if source_query is not None:
        # Treat as SQL query
        sql_query = source_query.strip()
        if not sql_query.lower().startswith('select'):
            return {"error": "Invalid source_query: Must start with SELECT."}
        
        try:
            # Execute the query using the RAG system's db_manager
            query_results = rag_system.db_manager.execute_query(sql_query)

            if not query_results:
                return {"error": f"Source query returned no results: {sql_query}"}

            # Check if the result has exactly one column
            if query_results and len(query_results[0]) != 1:
                 return {"error": f"Source query must return exactly one column. Query: {sql_query}"}

            # Extract the single column results
            final_source_names = [row[0] for row in query_results if row and row[0]]

            if not final_source_names:
                 return {"error": f"Source query returned no valid names: {sql_query}"}

        except (psycopg2.Error, Exception) as e:
            # Handle potential database errors
             return {"error": f"Error executing source_query '{sql_query}': {str(e)}"}

    elif source_names is not None:
        # Treat as a list of names, clean them
        if not isinstance(source_names, list):
             return {"error": "Invalid type for source_names. Must be a list of strings."}
        final_source_names = [name.strip().replace(",", "") for name in source_names if isinstance(name, str) and name.strip()]
    # else case is handled by the initial validation

    if not final_source_names:
        return {"error": "No valid source names provided or found from the input."}

    # Prepare arguments for rag_system.query
    query_args = {
        "user_query": keywords,
        "source_names": final_source_names,
        "get_children": get_children
    }
    if max_results_per_source is not None:
        query_args["max_results_per_source"] = max_results_per_source
        
    # Query the RAG system
    result = rag_system.query(**query_args)

    # The RAG system now returns the final structured output or an error message
    return result


def query_rag_from_id_func(
    block_indices: Union[int, List[int]],  # Block indices to retrieve
    source_name: Optional[str] = None,  # Document name (optional)
    get_children: bool = True,  # Whether to retrieve child blocks
    get_surrounding: bool = True  # Whether to retrieve surrounding blocks (2 before, 2 after)
    ) -> str:
    """
    Get specific document blocks by their indices, optionally including surrounding blocks.
    
    Args:
        block_indices: Block index or list of block indices to retrieve.
        source_name: Name of the document (optional).
        get_children: Whether to retrieve child blocks.
        get_surrounding: Whether to retrieve the 2 blocks before and 2 after each specified index.
        
    Returns:
        JSON string with text blocks information.
    """
    # Convert single index to list if needed
    if isinstance(block_indices, (int, str)):
        try:
            # Try to parse as JSON if it's a string representation of a list
            if isinstance(block_indices, str):
                block_indices = json.loads(block_indices)
            else:
                # If not JSON, treat as a single index
                block_indices = [block_indices]
        except:
            # If parsing fails, convert to list
            block_indices = [int(block_indices)]

    # Determine the final list of indices to fetch
    if get_surrounding:
        final_indices_to_fetch = set()
        for idx in block_indices:
            # Add the index itself and 2 before/after, ensuring non-negative
            for offset in range(-2, 3): # -2, -1, 0, 1, 2
                surrounding_idx = idx + offset
                if surrounding_idx >= 0:
                    final_indices_to_fetch.add(surrounding_idx)
        # Convert set to sorted list for consistent ordering
        indices_to_fetch = sorted(list(final_indices_to_fetch))
    else:
        # If not getting surrounding, just use the original list
        indices_to_fetch = block_indices

    # Get blocks by the potentially expanded list of indices
    blocks = rag_system.get_blocks_by_idx(indices_to_fetch, source_name, get_children)
    
    if not blocks:
        return {"error": "No blocks found with the provided indices"}
    
    # Extract text from blocks
    context_text = ""
    for block in blocks:
        if block["tag"] == "header":
            context_text += f"\n## {block['content']}\n"
        elif block["tag"] == "list_item":
            context_text += f"- {block['content']}\n"
        else:
            context_text += f"{block['content']}\n\n"
    
    # Return blocks information
    return [
            {
                "idx": block["block_idx"],
                "content": block["content"],
                "page": block["page_idx"],
                "parent_idx": block["parent_idx"],
                "level": block["level"],
                "tag": block["tag"]
            }
            for block in blocks
        ]


def highlight_pdf_func(
        pdf_file: str,
        block_indices: Optional[List[int]] = None,  # List of block indices to highlight
        debug: Optional[bool] = False
        ) -> Union[str, Dict[str, Any]]: 
    """
    Prepare information for highlighting a PDF by block indices. Checks if the PDF exists in the RAG system first.
    
    Args:
        pdf_file: Name of the PDF file (without path or extension)
        block_indices: List of block indices to highlight
        debug: Display all the blocks in the PDF for debugging (optional). Overwrite the parameter block_indices.
    
    Returns:
        JSON string containing PDF name and block indices for highlighting, or an error dictionary if the PDF is not found or multiple similar PDFs are found.
    """
    check_query = f"SELECT DISTINCT name FROM {schema_app_data}.rag_document_blocks WHERE name = %s LIMIT 1"
    exact_match = rag_system.db_manager.execute_query(check_query, (pdf_file,))

    found_pdf_name = None
    if exact_match:
        found_pdf_name = exact_match[0][0]
    else:
        cleaned_pdf_file = pdf_file.strip().replace(",", "")
        like_query = f"SELECT DISTINCT name FROM {schema_app_data}.rag_document_blocks WHERE name ILIKE %s"
        similar_matches = rag_system.db_manager.execute_query(like_query, (f"%{cleaned_pdf_file}%",))

        if not similar_matches:
            return {"error": f"PDF '{pdf_file}' not found in the RAG system (no exact or similar matches)."}
        elif len(similar_matches) == 1:
            found_pdf_name = similar_matches[0][0]
            print(f"Exact match for '{pdf_file}' not found. Using similar match: '{found_pdf_name}'")
        else:
            possible_names = [match[0] for match in similar_matches]
            return {"error": f"PDF '{pdf_file}' not found. Multiple similar documents exist: {', '.join(possible_names)}"}

    if found_pdf_name:
        result = {
            "pdf_file": found_pdf_name,
            "block_indices": block_indices,
            "debug": debug
        }
        return result
    else:
        return {"error": f"An unexpected error occurred while searching for PDF '{pdf_file}'."}

query_rag: BaseTool = tool(query_rag_func)
query_rag.name = "Query_RAG"
query_rag.description = """
Use this tool to search for information in documents.
Input is a question string.
The tool returns list of objects containing:
    - document name
    - id of the block
    - text content of the block
    - page number of the block
    - parent block id
    - level of the block
    - tag of the block (header, list_item, etc.)
"""

query_rag_from_id: BaseTool = tool(query_rag_from_id_func)
query_rag_from_id.name = "Query_RAG_From_Id"
query_rag_from_id.description = """
Use this tool to retrieve specific document blocks by their indices.
Input is a block index or list of block indices, and optionally a document name and whether to get children.
Use this to navigate through a document when you need to see additional blocks beyond the initial search results.
"""

highlight_pdf: BaseTool = tool(highlight_pdf_func)
highlight_pdf.name = "PDF_Viewer"
highlight_pdf.description = """
Use this tool to display a PDF with highlights.
Input is the PDF filename (without extension) and a list of block indices to highlight.
This will display the PDF with the specified blocks highlighted.
ATTENTION : The uploaded PDFs cannot be used in this tool. Only the PDFs in the RAG system can be used.
"""
