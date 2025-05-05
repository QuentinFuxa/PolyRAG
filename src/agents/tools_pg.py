import psycopg2
from langchain_core.tools import BaseTool, tool
import os
import json
from datetime import date, datetime

db_url = os.getenv("DATABASE_URL")

connection = psycopg2.connect(db_url)
cursor = connection.cursor()

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def execute_sql_func(sql_query: str) -> str:
    """Execute a read-only SQL query with safety checks

    Args:
        sql_query (str): A valid SQL query

    Returns:
        str: The result of the SQL query
    """
    sql_lower = sql_query.lower().strip()
    if not sql_lower.startswith('select'):
        return json.dumps({"error": "Only SELECT queries are allowed for security reasons."})
    
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        return json.dumps({"error": "Query contains potentially harmful operations."})
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        
        results_as_dict = []
        current_char_count = 2 # Start with 2 for initial '[]' or '{}' structure
        max_chars = 500
        warning_message = None
        original_row_count = len(results)

        for i, row in enumerate(results):
            row_dict = dict(zip(colnames, row))
            # Estimate the length added by this row's JSON representation
            # Add 1 for comma if not the first element
            row_json_len = len(json.dumps(row_dict, default=json_serial)) + (1 if i > 0 else 0) 
            
            if current_char_count + row_json_len > max_chars:
                warning_message = f"Result truncated: Query returned {original_row_count} rows, but the output was truncated to approximately {max_chars} characters ({len(results_as_dict)} rows shown)."
                break # Stop adding rows

            results_as_dict.append(row_dict)
            current_char_count += row_json_len

        # Prepare final output structure
        output_data = {"results": results_as_dict}
        if warning_message:
            output_data["warning"] = warning_message
            
        # Final check: if even the structure with warning is too long, return only error/warning
        final_json_output = json.dumps(output_data, default=json_serial)
        if len(final_json_output) > max_chars + 500: # Add some buffer for the warning itself
             return json.dumps({"warning": warning_message, "error": "Truncated result still exceeds character limit. Please refine your query."})

        return final_json_output
    except Exception as e:
        connection.rollback()  # Rollback the transaction on error
        error_message = str(e)
        return json.dumps({"error": f"Error executing query: {error_message}"})

execute_sql: BaseTool = tool(execute_sql_func)
execute_sql.name = "SQL_Executor"
