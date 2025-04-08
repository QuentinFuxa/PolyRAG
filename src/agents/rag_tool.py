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
    source_names: List[str], # Name of the letter(s) to search in. Example: ["INSSN-LYO-2023-0461"]
    get_children: bool = True, # Whether to retrieve child text blocks
    get_parents: bool = False, # Whether to retrieve parent text blocks
    search_strategy: SearchStrategyEnum = SearchStrategyEnum.TEXT
    ) -> str:
    """
    Query the RAG system to find relevant information in documents.
    
    Args:
        query: Words to search for in the documents. Example: ["ion", "circuit", "feu"]
        source_names: Name of the letter(s) to search in. Example: ["INSSN-LYO-2023-0461"]
        get_children: Whether to retrieve child text blocks
        get_parents: Whether to retrieve parent text blocks
        
    Returns:
        JSON string with query results and text blocks information
    """
    # Convert string enum to SearchStrategy
    strategy = SearchStrategy[search_strategy.upper()]
    
    # Query the RAG system
    result = rag_system.query(query, source_names=source_names, get_children=get_children, get_parents=get_parents, strategy=strategy)
    
    if not result["success"]:
        return {"error": result["message"]}
    
    # Return text blocks instead of annotations
    return (
            [{
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
    get_children: bool = False  # Whether to retrieve child blocks
    ) -> str:
    """
    Get specific document blocks by their indices
    
    Args:
        block_indices: Block index or list of block indices to retrieve
        source_name: Name of the document (optional)
        get_children: Whether to retrieve child blocks
        
    Returns:
        JSON string with text blocks information
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
    
    # Get blocks by indices
    blocks = rag_system.get_blocks_by_idx(block_indices, source_name, get_children)
    
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


def highlight_pdf_func(pdf_file: str, block_indices: List[int]) -> str:
    """
    Prepare information for highlighting a PDF by block indices.
    
    Args:
        pdf_file: Name of the PDF file (without path or extension)
        block_indices: List of block indices to highlight
    
    Returns:
        JSON string containing PDF name and block indices for highlighting
    """
    # Create response for display in streamlit frontend
    result = {
        "pdf_file": pdf_file,
        "block_indices": block_indices
    }
    return result


# Create tools
query_rag: BaseTool = tool(query_rag_func)
query_rag.name = "Query_RAG"
query_rag.description = """
Use this tool to search for information in documents.
Input is a question string and optionally a search strategy.
The tool returns text blocks related to the query.
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
"""