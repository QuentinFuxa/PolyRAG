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


# search = {"keywords":["cicatrices","chéloides"],"source_query":"SELECT name FROM public.public_data WHERE category ILIKE '%curiethérapie%'","count_only":True}

# result = rag_system.query(
#     user_query=["cicatrices","chéloides"],
#     source_query="SELECT name FROM public.public_data WHERE category ILIKE '%curiethérapie%'", 
#     count_only=False
# )

# result = rag_system.query(
#     user_query=["écarts"],
#     source_query="SELECT name FROM public.public_data WHERE EXTRACT(YEAR FROM sent_date) = 2024 AND (domains ILIKE '%curiethérapie%' OR category ILIKE '%curiethérapie%')", 
#     content_type="synthesis",
#     count_only=True,
# )


# result = rag_system.query(
#     user_query=['Radioprotection'],
#     source_query="SELECT name FROM public.public_data WHERE 'LUDD' = ANY(sector) AND EXTRACT(YEAR FROM sent_date) = 2024", 
#     count_only=True,
# )

# {"keywords":["radioprotection", "médical", "pratiques", "interventionnelles"] ,"source_query":"SELECT name FROM public.public_data","content_type":"demand","demand_priority":1} 


# result = rag_system.query(
#     user_query=["radioprotection", "médical", "pratiques", "interventionnelles"],
#     source_query="SELECT name FROM public.public_data", 
#     content_type="demand",
#     demand_priority=1
# )

# result = rag_system.query(
#     user_query=[],
#     source_query="SELECT name FROM public.public_data WHERE 'REP' = ANY(sector) AND theme ILIKE '%compétences%'",
#     content_type="demand",
# )

result = rag_system.query(
    user_query=["insomnia"],
    source_query="SELECT title FROM public.medrxiv_2025 ORDER BY publication_date DESC LIMIT 1",
    get_children=True
)




print(result)