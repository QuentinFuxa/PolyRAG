import json
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import os
from langchain_core.tools import BaseTool, tool

from rag_system import RAGSystem, SearchStrategy

# Initialize the RAG system
# Check if embeddings should be used based on environment variable
use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() in ("true", "1", "yes")
rag_system = RAGSystem(use_embeddings=use_embeddings)

class SearchStrategyEnum(str, Enum):
    TEXT = "text"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


def query_rag_func(
    query: List[str], # Words to search for in the documents. Example: ["ion", "circuit", "feu"]
    source_names: List[str], # Name of the letter(s) to search in. Example: ["INSSN-LYO-2023-0461", "INSNP-LYO-2016-0614"]
    get_children: bool = True, # Whether to retrieve child text blocks
    get_parents: bool = False, # Whether to retrieve parent text blocks
    search_strategy: SearchStrategyEnum = SearchStrategyEnum.TEXT
    ) -> str:
    """
    Query the RAG system to find relevant information in documents.
    
    Args:
        query: Words to search for in the documents. Example: ["ion", "circuit", "feu"]
        source_names: Name of the letter(s) to search in. Example: ["INSSN-LYO-2023-0461", "INSNP-LYO-2016-0614"]
        get_children: Whether to retrieve child text blocks
        get_parents: Whether to retrieve parent text blocks
        
    Returns:
        JSON string with, for each block:
            - document name
            - id of the block
            - text content of the block
            - page number of the block
            - parent block id
            - level of the block
            - tag of the block (header, list_item, etc.)

    """
    # Convert string enum to SearchStrategy
    strategy = SearchStrategy[search_strategy.upper()]
    
    # clean source_names to remove any leading/trailing whitespace or commas
    source_names = [name.strip().replace(",", "") for name in source_names if name.strip()]
    
    # Query the RAG system
    result = rag_system.query(
        query,
        source_names=source_names,
        get_children=get_children,
        get_parents=get_parents,
        strategy=strategy,
        num_results = 15
        )
    
    if not result["success"]:
        return {"error": result["message"]}
    
    # Return text blocks instead of annotations
    return (
            [{  
                "document_name": block["name"],
                "idx": block["block_idx"],  # Using block_idx instead of block_id
                "content": block["content"],
                "page": block["page_idx"],
                "parent_idx": block["parent_idx"],  # Using parent_idx instead of parent_id
                "level": block["level"],
                "tag": block["tag"]
            }
            for block in result["context_blocks"]]
        )


def query_rag_from_idx_func(
    block_indices: Union[int, List[int]],  # Block indices to retrieve
    source_name: Optional[str] = None,  # Document name (optional)
    get_children: bool = False,  # Whether to retrieve child blocks
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
        ) -> str:
    """
    Prepare information for highlighting a PDF by block indices.
    
    Args:
        pdf_file: Name of the PDF file (without path or extension)
        block_indices: List of block indices to highlight
        debug: Display all the blocks in the PDF for debugging (optional). Overwrite the parameter block_indices.
    
    Returns:
        JSON string containing PDF name and block indices for highlighting
    """
    # Create response for display in streamlit frontend
    result = {
        "pdf_file": pdf_file,
        "block_indices": block_indices,
        "debug": debug
    }
    return result


# Create tools
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

query_rag_from_idx: BaseTool = tool(query_rag_from_idx_func)
query_rag_from_idx.name = "Query_RAG_From_Idx"
query_rag_from_idx.description = """
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
