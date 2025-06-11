import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agents.tools_pg import execute_sql_func


debug_params = {"sql_query":"SELECT COUNT(*) as count FROM public.public_data WHERE type_inspect = 'C : courante' AND sent_date < '2020-01-01'"}


result = execute_sql_func(**debug_params)

if isinstance(result, (list, dict)):
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(result)