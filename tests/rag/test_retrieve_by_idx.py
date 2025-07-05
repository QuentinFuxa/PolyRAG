import json
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import os
import psycopg2 # Import psycopg2 for error handling
from langchain_core.tools import BaseTool, tool

from agents.rag_tool import query_rag_from_id_func

result = query_rag_from_id_func(
    block_indices = [107],
    source_name = "INSNP-PRS-2025-0869"
)
print(result)