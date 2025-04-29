import psycopg2
from langchain_core.tools import BaseTool, tool
import os

db_url = os.getenv("DATABASE_URL")

connection = psycopg2.connect(db_url)
cursor = connection.cursor()   

def execute_sql_func(sql_query: str) -> str:
    """Execute a read-only SQL query with safety checks

    Args:
        sql_query (str): A valid SQL query

    Returns:
        str: The result of the SQL query
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
        return str(results)
    except Exception as e:
        return f"Error executing query: Invalid query syntax or permissions"

execute_sql: BaseTool = tool(execute_sql_func)
execute_sql.name = "SQL_Executor"
