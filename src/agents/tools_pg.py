import psycopg2
from langchain_core.tools import BaseTool, tool
connection_string="postgresql://postgres@localhost:5432/lds"

connection = psycopg2.connect(connection_string)
cursor = connection.cursor()   

def execute_sql_func(sql_query: str) -> str:
    """Execute an SQL query

    Args:
        sql_query (str): A valid SQL query

    Returns:
        str: The result of the SQL query
    """

    cursor.execute(sql_query)
    results = cursor.fetchall()
    return str(results)

execute_sql: BaseTool = tool(execute_sql_func)
execute_sql.name = "SQL_Executor"
