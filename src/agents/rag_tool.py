import json
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import os
import psycopg2 # Import psycopg2 for error handling
from langchain_core.tools import BaseTool, tool

from rag_system import RAGSystem
from db_manager import schema_app_data # Import schema_app_data

# Initialize the RAG system
rag_system = RAGSystem()


def query_rag_func(
    keywords: List[str],
    source_query: Optional[str] = None,
    source_names: Optional[List[str]] = None,
    get_children: bool = True,
    offset: int = 0,
    content_type: Optional[str] = None,
    section_filter: Optional[List[str]] = None,
    demand_priority: Optional[int] = None,
    count_only: bool = False
    ) -> Union[Dict[str, Any], str]:
    """
    Query the RAG system to find relevant information in documents with simplified return format and single-query source handling.

    Args:
        keywords: Words to search for in the documents. Example: ["télécommunications", "crise"].
        source_query: A SQL query string that returns document names. Example: "SELECT name FROM public.public_data ORDER BY sent_date DESC LIMIT 1". Use this OR source_names, or leave both None to search all documents.
        source_names: A list of document names to search within. Example: ["report_campaign", "biology_comparison_report"]. Use this OR source_query, or leave both None to search all documents.
        get_children: Whether to retrieve child blocks for each found block. Defaults to True.
        offset: Number of results to skip for pagination (default: 0). Results contains max 20 results. If you want to get the next 20 results, use offset=20.
        content_type: Filter by content type: 'demand', 'section_header', or 'regular'
        section_filter: Filter by section types: e.g., ['synthesis', 'demands', 'observations']
        demand_priority: Filter demands by priority: 1 (prioritaires) or 2 (complémentaires)
        count_only: If True, return only the count of matching blocks instead of the content.

    Returns:
        Dictionary containing:
            - total_number_results (int): Total number of matching results.
            - number_returned_results (int): Number of results in this response.
            - results (list): List of result blocks.
        Or an error dictionary if something goes wrong.
    """
    if source_names is not None and source_query is not None:
        return {"error": "Provide either 'source_names' (list of strings) or 'source_query' (SQL string), but not both."}

    if source_query and not source_query.strip().lower().startswith('select'):
        return {"error": "Invalid source_query: Must start with SELECT if provided."}

    try:
        result = rag_system.query(
            user_query=keywords,
            source_query=source_query, 
            source_names=source_names,
            offset=offset,
            get_children=get_children,
            content_type=content_type,
            section_filter=section_filter,
            demand_priority=demand_priority,
            count_only=count_only
        )
        return result
    except Exception as e:
        return {"error": f"Error during search: {str(e)}"}


def query_rag_from_id_func(
    block_indices: Union[int, List[int]],
    source_name: Optional[str] = None,
    get_children: bool = True,
    get_surrounding: bool = True
    ) -> List[Dict[str, Any]]: # Return type changed to List[Dict]
    """
    Get specific document blocks by their indices, optionally including surrounding blocks.
    
    Args:
        block_indices: Block index or list of block indices to retrieve.
        source_name: Name of the document (optional).
        get_children: Whether to retrieve child blocks.
        get_surrounding: Whether to retrieve the 2 blocks before and 2 after each specified index.
        
    Returns:
        List of dictionaries with text blocks information, or a list containing an error dictionary.
    """
    # Convert single index to list if needed
    if isinstance(block_indices, (int, str)):
        try:
            if isinstance(block_indices, str):
                block_indices = json.loads(block_indices)
            else:
                block_indices = [block_indices]
        except (json.JSONDecodeError, TypeError, ValueError): # Catch specific errors
            try:
                block_indices = [int(block_indices)]
            except ValueError:
                 return [{"error": "Invalid block_indices format. Must be an int, list of ints, or JSON string of ints."}]


    # Determine the final list of indices to fetch
    final_indices_to_fetch = set()
    if get_surrounding:
        for idx_val in block_indices: # Use a different variable name for the loop
            try:
                current_idx = int(idx_val)
                for offset_val in range(-2, 3): 
                    surrounding_idx = current_idx + offset_val
                    if surrounding_idx >= 0:
                        final_indices_to_fetch.add(surrounding_idx)
            except ValueError:
                return [{"error": f"Invalid index '{idx_val}' in block_indices."}]
        indices_to_fetch = sorted(list(final_indices_to_fetch))
    else:
        try:
            indices_to_fetch = [int(idx_val) for idx_val in block_indices]
        except ValueError:
            return [{"error": "Invalid index in block_indices. All must be integers."}]


    # Get blocks by the potentially expanded list of indices
    blocks = rag_system.get_blocks_by_idx(indices_to_fetch, source_name, get_children)
    
    if not blocks: # get_blocks_by_idx should return a list
        return [{"error": "No blocks found with the provided indices"}] # Return list with error dict
    
    # Try to enrich blocks with classification metadata
    try:
        # Ensure blocks is a list of dicts and block_idx exists
        block_indices_list = [block["block_idx"] for block in blocks if isinstance(block, dict) and "block_idx" in block]
        if block_indices_list:
            metadata_query = f"""
            SELECT 
                block_idx,
                content_type,
                section_type,
                demand_priority
            FROM 
                {schema_app_data}.rag_document_blocks
            WHERE 
                block_idx IN ({', '.join(['%s'] * len(block_indices_list))})
            """
            
            metadata_results = rag_system.db_manager.execute_query(
                metadata_query, tuple(block_indices_list)
            )
            
            metadata_lookup = {}
            if metadata_results:
                for row in metadata_results:
                    if row[1] or row[2] or row[3]:
                        metadata_lookup[row[0]] = {
                            "content_type": row[1],
                            "section_type": row[2],
                            "demand_priority": row[3]
                        }
            
            for block in blocks:
                if isinstance(block, dict) and block.get("block_idx") in metadata_lookup:
                    block.update(metadata_lookup[block["block_idx"]])
    except Exception:
        # Silently continue if enrichment fails, or log an error
        pass 
    
    # Return enriched blocks information
    result_blocks = []
    for block in blocks:
        if not isinstance(block, dict): continue # Skip if block is not a dict

        block_info = {
            "idx": block.get("block_idx"),
            "content": block.get("content"),
            "parent_idx": block.get("parent_idx"),
            "level": block.get("level"),
            "tag": block.get("tag")
        }
        # Add classification metadata if present
        if "content_type" in block:
            block_info["content_type"] = block["content_type"]
        if "section_type" in block:
            block_info["section_type"] = block["section_type"]
        if "demand_priority" in block:
            block_info["demand_priority"] = block["demand_priority"]
        
        result_blocks.append(block_info)
    
    return result_blocks


def highlight_pdf_func(
        pdf_requests: List[Dict[str, Any]],
        debug: Optional[bool] = False
        ) -> str:
    """
    Prepare information for highlighting multiple PDFs by block indices.
    
    Args:
        pdf_requests: A list of dictionaries, where each dictionary contains:
                      - "pdf_file": Name of the PDF file (without path or extension).
                      - "block_indices": List of block indices to highlight for this PDF.
        debug: Display all the blocks in all requested PDFs for debugging (optional).
               This flag applies globally to all PDFs in the request.
    
    Returns:
        A JSON string representing a list of results. Each item in the list is either:
        - A success dictionary: {"pdf_file": str, "block_indices": List[int], "debug": bool}
        - An error dictionary: {"error": str, "original_request": Dict[str, Any]}
    """
    results = []
    for request in pdf_requests:
        pdf_file = request.get("pdf_file")
        block_indices = request.get("block_indices")

        if not pdf_file or not isinstance(pdf_file, str):
            results.append({"error": "Missing or invalid 'pdf_file' in request.", "original_request": request})
            continue

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
                results.append({"error": f"PDF '{pdf_file}' not found in the RAG system (no exact or similar matches).", "original_request": request})
                continue
            elif len(similar_matches) == 1:
                found_pdf_name = similar_matches[0][0]
                print(f"Exact match for '{pdf_file}' not found. Using similar match: '{found_pdf_name}'")
            else:
                possible_names = [match[0] for match in similar_matches]
                results.append({"error": f"PDF '{pdf_file}' not found. Multiple similar documents exist: {', '.join(possible_names)}", "original_request": request})
                continue
        
        if found_pdf_name:
            results.append({
                "pdf_file": found_pdf_name,
                "block_indices": block_indices,
                "debug": debug  # Use the global debug flag
            })
        # No explicit else needed here as errors are handled by `continue` statements above.

    return json.dumps(results)

query_rag: BaseTool = tool(query_rag_func)
query_rag.name = "Query_RAG"
query_rag.description = """
Use this tool to search for information in documents with simplified return format and pagination support.
Enhanced with demand classification and single-query source handling.

Parameters:
- keywords: List of keywords to search (e.g., ["télécommunications", "crise"])
- source_query: SQL query returning document names (e.g., "SELECT name FROM public.public_data ORDER BY sent_date DESC LIMIT 1")
- source_names: List of document names OR use source_query (not both)
- limit: Maximum number of results to return (default: 20)
- offset: Number of results to skip for pagination (default: 0)
- content_type: Optional filter ('demand', 'section_header', 'regular')
- section_filter: Optional filter by sections (['synthesis', 'demands', 'observations', etc.])
- demand_priority: Optional filter (1 for prioritaires, 2 for complémentaires)
- count_only: If True, returns statistics instead of content (currently provides total_count only when source_query is used).

Returns (if count_only=False):
    - total_number_results: Total number of matching results
    - number_returned_results: Number of results in this response
    - results: List of result blocks with metadata (document_name, idx, content, level, tag, content_type, section_type, demand_priority, children)

Returns (if count_only=True):
    - total_count: Total number of matching blocks.
    - (Note: Detailed breakdown by document/section/priority with source_query is simplified in this version.)

For pagination, use offset parameter. Example: offset=20 to get next 20 results.
The source_query now integrates directly with the search, solving the issue where keywords might not appear in the filtered documents.
"""

query_rag_from_id: BaseTool = tool(query_rag_from_id_func)
query_rag_from_id.name = "Query_RAG_From_Id"
query_rag_from_id.description = """
Use this tool to retrieve specific document blocks by their indices.
Input is a block index or list of block indices, and optionally a document name and whether to get children.
Removes page_idx from output.
Use this to navigate through a document when you need to see additional blocks beyond the initial search results.
"""

highlight_pdf: BaseTool = tool(highlight_pdf_func)
highlight_pdf.name = "PDF_Viewer"
highlight_pdf.description = """
Use this tool to display one or more PDFs with highlights.
Input consists of a list of PDF requests and an optional global debug flag.
Each PDF request in the list should be a dictionary specifying "pdf_file" (name of the PDF, without path or extension)
and "block_indices" (list of integer block indices to highlight for that PDF).
The "debug" flag, if true, applies to all PDFs in the request, causing all their blocks to be highlighted.
This tool will prepare the information for the UI to display buttons for each PDF, allowing users to view them with the specified highlights.
ATTENTION : The uploaded PDFs cannot be used in this tool. Only the PDFs in the RAG system can be used.

Example `pdf_requests` parameter:
[
    {"pdf_file": "document_A_name", "block_indices": [10, 15, 22]},
    {"pdf_file": "document_B_name", "block_indices": [5, 8]}
]
"""
