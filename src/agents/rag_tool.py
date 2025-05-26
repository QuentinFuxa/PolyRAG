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
    max_results_per_source: Optional[int] = None, # Maximum number of results to return per source document. Defaults to 3 in the RAG system.
    content_type: Optional[str] = None, # Filter by content type: 'demand', 'section_header', or 'regular'
    section_filter: Optional[List[str]] = None, # Filter by section types: e.g., ['synthesis', 'demands', 'observations']
    demand_priority: Optional[int] = None, # Filter demands by priority: 1 (prioritaires) or 2 (complémentaires)
    count_only: bool = False # If True, return only statistics instead of full results
    ) -> Union[Dict[str, Any], str]:
    """
    Query the RAG system to find relevant information in documents. Provide EITHER source_names OR source_query.

    Args:
        keywords: Words to search for in the documents. Example: ["isotope & atoms", "molecule", "reaction"]. Use "&" to require to have both words (or more) in the result.
        source_query: A SQL query string that returns a single column containing document names. Example: "SELECT doc_name FROM documents WHERE year = 2025". The query MUST start with SELECT and return only one column. Use this OR source_names.
        source_names: A list of document names to search within. Example: ["report_campaign", "biology_comparison_report"]. Use this OR source_query.
        get_children: Whether to retrieve child blocks for each found block. Defaults to True.
        max_results_per_source: Maximum number of results to return per source document. If None, defaults to 3.
        content_type: Filter by content type: 'demand' (demandes), 'section_header' (titres de sections), or 'regular' (contenu normal)
        section_filter: Filter by section types. Valid values: 'synthesis', 'demands', 'demandes_prioritaires', 'autres_demandes', 'information', 'observations'
        demand_priority: Filter demands by priority: 1 for "demandes prioritaires", 2 for "demandes complémentaires"
        count_only: If True, return only count statistics instead of full results

    Returns:
        If count_only is False:
            The structured result from the RAG system with enriched metadata including:
                - document name
                - id of the block
                - text content of the block
                - page number of the block
                - parent block id
                - level of the block
                - tag of the block (header, list_item, etc.)
                - content_type (if classified)
                - section_type (if classified)
                - demand_priority (if it's a demand)
        If count_only is True:
            Statistics dictionary containing:
                - total_count: Total number of matching blocks
                - by_document: Count breakdown by document
                - by_section: Count breakdown by section (if applicable)
                - by_priority: Count breakdown by priority (if applicable)
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

    # If count_only is True, perform counting logic
    if count_only:
        try:
            # Build count query with filters
            count_query = f"""
            SELECT 
                name,
                section_type,
                demand_priority,
                COUNT(*) as count
            FROM 
                {schema_app_data}.rag_document_blocks
            WHERE 
                name IN ({', '.join(['%s'] * len(final_source_names))})
            """
            
            params = final_source_names.copy()
            conditions = []
            
            # Add content type filter
            if content_type:
                conditions.append("content_type = %s")
                params.append(content_type)
            
            # Add section filter
            if section_filter:
                placeholders = ', '.join(['%s'] * len(section_filter))
                conditions.append(f"section_type IN ({placeholders})")
                params.extend(section_filter)
            
            # Add demand priority filter
            if demand_priority:
                conditions.append("demand_priority = %s")
                params.append(demand_priority)
            
            # Add keyword search if provided
            if keywords:
                ts_query = " & ".join([kw.replace(" & ", " ") for kw in keywords])
                conditions.append(f"content_tsv @@ to_tsquery('{rag_system.TS_QUERY_LANGUAGE}', %s)")
                params.append(ts_query)
            
            if conditions:
                count_query += " AND " + " AND ".join(conditions)
            
            count_query += " GROUP BY name, section_type, demand_priority"
            count_query += " ORDER BY name, demand_priority, section_type"
            
            results = rag_system.db_manager.execute_query(count_query, tuple(params))
            
            # Process results into statistics
            statistics = {
                "total_count": 0,
                "by_document": {},
                "by_section": {},
                "by_priority": {}
            }
            
            for row in results:
                doc_name = row[0]
                section_type = row[1]
                priority = row[2]
                count = row[3]
                
                statistics["total_count"] += count
                
                # Count by document
                if doc_name not in statistics["by_document"]:
                    statistics["by_document"][doc_name] = 0
                statistics["by_document"][doc_name] += count
                
                # Count by section
                if section_type:
                    if section_type not in statistics["by_section"]:
                        statistics["by_section"][section_type] = 0
                    statistics["by_section"][section_type] += count
                
                # Count by priority
                if priority:
                    priority_label = f"priority_{priority}"
                    if priority_label not in statistics["by_priority"]:
                        statistics["by_priority"][priority_label] = 0
                    statistics["by_priority"][priority_label] += count
            
            return statistics
            
        except Exception as e:
            return {"error": f"Error during counting: {str(e)}"}
    
    # For regular search, pass filter parameters directly to RAG system
    query_args = {
        "user_query": keywords,
        "source_names": final_source_names,
        "get_children": get_children,
        "content_type": content_type,
        "section_filter": section_filter,
        "demand_priority": demand_priority
    }
    if max_results_per_source is not None:
        query_args["max_results_per_source"] = max_results_per_source
        
    # Query the RAG system with filters - it will handle them efficiently at the DB level
    # The RAG system now returns already filtered results with metadata included directly from the search
    result = rag_system.query(**query_args)
    
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
    
    # Try to enrich blocks with classification metadata
    try:
        block_indices_list = [block["block_idx"] for block in blocks]
        if block_indices_list:
            # Query for metadata
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
            
            # Create metadata lookup
            metadata_lookup = {}
            for row in metadata_results:
                if row[1] or row[2] or row[3]:  # Only add if there's some metadata
                    metadata_lookup[row[0]] = {
                        "content_type": row[1],
                        "section_type": row[2],
                        "demand_priority": row[3]
                    }
            
            # Enrich blocks
            for block in blocks:
                if block["block_idx"] in metadata_lookup:
                    block.update(metadata_lookup[block["block_idx"]])
    except Exception:
        # Silently continue if enrichment fails
        pass
    
    # Extract text from blocks
    context_text = ""
    for block in blocks:
        if block["tag"] == "header":
            context_text += f"\n## {block['content']}\n"
        elif block["tag"] == "list_item":
            context_text += f"- {block['content']}\n"
        else:
            context_text += f"{block['content']}\n\n"
    
    # Return enriched blocks information
    result_blocks = []
    for block in blocks:
        block_info = {
            "idx": block["block_idx"],
            "content": block["content"],
            "page": block["page_idx"],
            "parent_idx": block["parent_idx"],
            "level": block["level"],
            "tag": block["tag"]
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
Enhanced with demand classification: can filter by content type, section, and demand priority.
Can also return statistics only (count_only mode).

Parameters:
- keywords: List of keywords to search (e.g., ["incendie", "risque"])
- source_query or source_names: SQL query or list of document names
- content_type: Optional filter ('demand', 'section_header', 'regular')
- section_filter: Optional filter by sections (['synthesis', 'demands', 'observations', etc.])
- demand_priority: Optional filter (1 for prioritaires, 2 for complémentaires)
- count_only: If True, returns statistics instead of content

Returns (if count_only=False):
    - document name
    - id of the block
    - text content of the block
    - page number of the block
    - parent block id
    - level of the block
    - tag of the block (header, list_item, etc.)
    - content_type (if classified)
    - section_type (if classified)
    - demand_priority (if it's a demand)

Returns (if count_only=True):
    - total_count
    - by_document breakdown
    - by_section breakdown
    - by_priority breakdown
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
