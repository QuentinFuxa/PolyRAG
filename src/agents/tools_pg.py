import psycopg2
from langchain_core.tools import BaseTool, tool
import os
import csv
import io 
from datetime import date, datetime
from decimal import Decimal
db_url = os.getenv("DATABASE_URL")

connection = psycopg2.connect(db_url)
cursor = connection.cursor()


def to_csv_string_value(obj):
    """Converts Python objects to string representations suitable for CSV cells."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(float(obj))
    if obj is None:
        return ""
    return str(obj)

def execute_sql_func(sql_query: str) -> str:
    """Execute a read-only SQL query with safety checks and returns results as CSV.

    Args:
        sql_query (str): A valid SQL query

    Returns:
        str: The result of the SQL query as a CSV string (semicolon separated),
             or an error message if the query fails or is disallowed.
    """
    sql_lower = sql_query.lower().strip()
    if not sql_lower.startswith('select'):
        return "Error: Only SELECT queries are allowed for security reasons."
    
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        return "Error: Query contains potentially harmful operations."
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        if not cursor.description:
            if results:
                output_io = io.StringIO()
                csv_writer = csv.writer(output_io, delimiter=';')
                if len(results[0]) == 1:
                    csv_writer.writerow(["result"])
                else:
                    csv_writer.writerow([f"col_{i+1}" for i in range(len(results[0]))])
                for row_data in results:
                    csv_writer.writerow([to_csv_string_value(item) for item in row_data])
                return output_io.getvalue()
            return ""

        colnames = [desc[0] for desc in cursor.description]
        
        output_io = io.StringIO()
        csv_writer = csv.writer(output_io, delimiter=';')
        
        csv_writer.writerow(colnames)
        
        max_rows = 500
        warning_message_str = None
        original_row_count = len(results)

        for i, row_data in enumerate(results):
            if i >= max_rows:
                warning_message_str = f"# Warning: Query returned {original_row_count} rows, but the output was truncated to {max_rows} rows."
                break
            
            processed_row = [to_csv_string_value(item) for item in row_data]
            csv_writer.writerow(processed_row)

        csv_output_string = output_io.getvalue()
        
        if warning_message_str:
            csv_output_string += warning_message_str + "\n"
            
        max_chars_overall = 15000
        if len(csv_output_string) > max_chars_overall:
            truncated_csv_string = csv_output_string[:max_chars_overall - 100]
            last_newline = truncated_csv_string.rfind('\n')
            if last_newline != -1:
                truncated_csv_string = truncated_csv_string[:last_newline+1]
            
            truncation_error_msg = f"# Error: Result too large. Output truncated to approx {max_chars_overall} chars.\n"
            if warning_message_str and warning_message_str in truncated_csv_string:
                 return truncated_csv_string + truncation_error_msg
            elif warning_message_str:
                return truncated_csv_string + warning_message_str + "\n" + truncation_error_msg
            else:
                return truncated_csv_string + truncation_error_msg

        return csv_output_string
    except Exception as e:
        connection.rollback()
        error_message = str(e).replace('\n', ' ').strip()
        return f"Error executing query: {error_message}"

execute_sql: BaseTool = tool(execute_sql_func)
execute_sql.name = "SQL_Executor"
