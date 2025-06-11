import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from agents.rag_tool import query_rag_func


debug_params= {"keywords":["fraude"],"source_query":"SELECT name FROM public.public_data WHERE EXTRACT(YEAR FROM sent_date) = 2024","count_only":True}

result = query_rag_func(**debug_params
)

if isinstance(result, (list, dict)):
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(result)