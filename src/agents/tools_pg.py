from langchain_core.tools import BaseTool, tool
import os
import csv
import io
from datetime import date, datetime
from decimal import Decimal
from typing import List, Literal, Optional, Union
import logging # Added for logging

from src.db_manager import DatabaseManager # Import DatabaseManager

logger = logging.getLogger(__name__) # Added logger

# Instantiate DatabaseManager - it's a singleton
try:
    db_manager = DatabaseManager()
except Exception as e:
    logger.error(f"Failed to initialize DatabaseManager in tools_pg: {e}")
    # Depending on desired behavior, could raise this or have tools fail gracefully
    db_manager = None 

def to_csv_string_value(obj):
    """Converts Python objects to string representations suitable for CSV cells."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(float(obj))
    if obj is None:
        return ""
    return str(obj)

def _execute_safe_query(query: str, params: tuple = None) -> List[tuple]:
    """Executes a read-only query safely using DatabaseManager."""
    if not db_manager:
        raise ConnectionError("DatabaseManager not initialized. Cannot execute query.")
    try:
        # DatabaseManager's execute_query handles connection and cursor management
        results = db_manager.execute_query(query, params)
        return results
    except Exception as e:
        # db_manager.execute_query should handle rollbacks if it initiated a transaction
        # Re-raise or handle as appropriate for the tool
        error_message = str(e).replace('\n', ' ').strip()
        logger.error(f"Error executing query via DatabaseManager: {error_message} SQL: {query} PARAMS: {params}")
        raise ValueError(f"Error executing query: {error_message}")

def _get_letter_names_from_subquery(subquery: str) -> List[str]:
    """Executes a subquery against public.public_data expected to return letter names."""
    sql_lower = subquery.lower().strip()
    # Basic safety check for subquery - MUST target public.public_data
    if not sql_lower.startswith('select'):
        raise ValueError("Error: Subquery must be a SELECT statement.")
    if 'from public.public_data' not in sql_lower:
        raise ValueError("Error: Subquery must target the 'public.public_data' table.")
    if not sql_lower.startswith('select'):
        raise ValueError("Error: Subquery must be a SELECT statement.")
    # Allow common filtering keywords but block modification keywords
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        raise ValueError("Error: Subquery contains potentially harmful operations.")

    try:
        results = _execute_safe_query(subquery)
        # Expecting a single column of strings (letter names)
        letter_names = [row[0] for row in results if isinstance(row[0], str)]
        if not letter_names:
             raise ValueError("Subquery did not return any valid letter names from public.public_data.")
        return letter_names
    except Exception as e:
        raise ValueError(f"Error executing subquery: {e}")


def _get_demands_data(
    return_type: Literal['content', 'count'],
    priority: Optional[int] = None, # Made priority optional
    letter_names: Optional[List[str]] = None,
    letter_name_subquery: Optional[str] = None
) -> Union[List[str], int]:
    """
    Core function to fetch demand data (content or count) based on optional priority
    and letter filters (names from public.public_data via list or subquery).
    If priority is None, fetches all demands (priority 1 and 2).
    """
    if not letter_names and not letter_name_subquery:
        raise ValueError("Either letter_names or letter_name_subquery must be provided.")
    if letter_names and letter_name_subquery:
        raise ValueError("Provide either letter_names or letter_name_subquery, not both.")

    target_letter_names = []
    if letter_names:
        # Validate input are strings
        if not all(isinstance(name, str) for name in letter_names):
             raise ValueError("letter_names must be a list of strings.")
        target_letter_names = letter_names
    elif letter_name_subquery:
        # Execute subquery against public.public_data to get names
        target_letter_names = _get_letter_names_from_subquery(letter_name_subquery)

    if not target_letter_names:
         return 0 if return_type == 'count' else [] # No letters to query

    # Use tuple for psycopg2 parameter substitution for names
    names_tuple = tuple(target_letter_names)
    print(target_letter_names)
    placeholders = ', '.join(['%s'] * len(names_tuple))

    # Build the base query parts
    select_clause = "COUNT(*)" if return_type == 'count' else "l.text, d.start, d.end"
    base_query = f"""
        SELECT {select_clause}
        FROM public.demands d
        JOIN public.letters l ON d.id_letter = l.id_letter
        WHERE l.name IN ({placeholders})
    """
    params_list = list(names_tuple) # Start params with letter names

    # Add priority filter conditionally
    if priority is not None:
        base_query += " AND d.priority = %s"
        params_list.append(priority) # Append priority to params list

    # Add ordering for content retrieval
    if return_type == 'content':
        base_query += " ORDER BY l.name, d.start"

    # Execute the query
    params = tuple(params_list)
    print(params)
    results = _execute_safe_query(base_query, params)

    # Process results
    if return_type == 'count':
        return results[0][0] if results else 0
    elif return_type == 'content':
        demand_texts = []
        for text, start, end in results:
            if text is not None and start is not None and end is not None:
                # Ensure start/end are within bounds
                start = max(0, start)
                end = min(len(text), end)
                if start < end:
                    demand_texts.append(text[start:end])
        return demand_texts
    else:
        raise ValueError("Invalid return_type specified.")


@tool
def get_demand_content(
    demand_type: Optional[Literal["Demandes à traiter prioritairement", "Autres demandes"]] = None, # Made optional
    letter_names: Optional[List[str]] = None,
    letter_name_subquery: Optional[str] = None
) -> List[str]:
    """
    Retrieves the text content of demands from letters identified by name.
    Requires either a list of letter names or a SQL subquery targeting public.public_data returning names.
    If demand_type is not specified, retrieves all demands (priority 1 and 2).

    Args:
        demand_type: (Optional) The type of demand ('Demandes à traiter prioritairement' or 'Autres demandes'). Defaults to all.
        letter_names: A list of exact letter names (strings) for the letters to search within.
        letter_name_subquery: A SQL SELECT query string targeting public.public_data that returns a list of letter names.
                              Example: "SELECT name FROM public.public_data WHERE site_name = 'Blayais'"

    Returns:
        A list of strings, where each string is the content of a matching demand.
        Returns an error message string on failure.
    """
    try:
        priority = None
        if demand_type:
            priority_map = {
                "Demandes à traiter prioritairement": 1,
                "Autres demandes": 2
            }
            priority = priority_map.get(demand_type)
            if priority is None:
                # This case should ideally not happen due to Literal typing, but good practice
                return ["Error: Invalid demand_type specified."]

        return _get_demands_data(
            return_type='content',
            priority=priority, # Pass None if demand_type wasn't specified
            letter_names=letter_names,
            letter_name_subquery=letter_name_subquery
        )
    except (ValueError, ConnectionError, Exception) as e:
        return [f"Error: {e}"]


@tool
def count_demands(
    demand_type: Optional[Literal["Demandes à traiter prioritairement", "Autres demandes"]] = None, # Made optional
    letter_names: Optional[List[str]] = None,
    letter_name_subquery: Optional[str] = None
) -> Union[int, str]:
    """
    Counts the number of demands within letters identified by name.
    Requires either a list of letter names or a SQL subquery targeting public.public_data returning names.
    If demand_type is not specified, counts all demands (priority 1 and 2).

    Args:
        demand_type: (Optional) The type of demand to count ('Demandes à traiter prioritairement' or 'Autres demandes'). Defaults to all.
        letter_names: A list of exact letter names (strings) for the letters to search within.
        letter_name_subquery: A SQL SELECT query string targeting public.public_data that returns a list of letter names.
                              Example: "SELECT name FROM public.public_data WHERE site_name = 'Blayais'"

    Returns:
        An integer representing the total count of matching demands.
        Returns an error message string on failure.
    """
    try:
        priority = None
        if demand_type:
            priority_map = {
                "Demandes à traiter prioritairement": 1,
                "Autres demandes": 2
            }
            priority = priority_map.get(demand_type)
            if priority is None:
                 # This case should ideally not happen due to Literal typing, but good practice
                return "Error: Invalid demand_type specified."

        return _get_demands_data(
            return_type='count',
            priority=priority, # Pass None if demand_type wasn't specified
            letter_names=letter_names,
            letter_name_subquery=letter_name_subquery
        )
    except (ValueError, ConnectionError, Exception) as e:
        return f"Error: {e}"


# --- Existing execute_sql function ---

def execute_sql_func(sql_query: str) -> str:
    """Execute a read-only SQL query with safety checks and returns results as CSV.

    Args:
        sql_query (str): A valid SQL query

    Returns:
        str: The result of the SQL query as a CSV string (semicolon separated),
             or an error message if the query fails or is disallowed.
    """
    if not db_manager:
        return "Error: DatabaseManager not initialized. Cannot execute SQL."

    sql_lower = sql_query.lower().strip()
    if not sql_lower.startswith('select'):
        return "Error: Only SELECT queries are allowed for security reasons."

    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        return "Error: Query contains potentially harmful operations."

    conn = None
    try:
        # We need direct cursor access for cursor.description to get column names.
        # So, we manage the connection lifecycle here.
        conn = db_manager.get_connection()
        with conn.cursor() as current_cursor: # Use 'current_cursor' to avoid conflict if 'cursor' was a global
            current_cursor.execute(sql_query)
            results = current_cursor.fetchall()

            # Try to get column names from cursor.description
            # This part remains largely the same, but uses current_cursor
            if not current_cursor.description: # Fallback if no description
                if results:
                    output_io = io.StringIO()
                    csv_writer = csv.writer(output_io, delimiter=';')
                    # Determine number of columns from the first row if possible
                    num_cols = len(results[0]) if results and isinstance(results[0], (list, tuple)) else 1
                    if num_cols == 1:
                        csv_writer.writerow(["result"])
                    else:
                        csv_writer.writerow([f"col_{i+1}" for i in range(num_cols)])
                    for row_data in results:
                        csv_writer.writerow([to_csv_string_value(item) for item in row_data])
                    return output_io.getvalue()
                return "" # No results and no description

            colnames = [desc[0] for desc in current_cursor.description]

        # Proceed with CSV generation using colnames and results
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
        
        # conn.commit() # Typically SELECT queries don't need commit unless they call functions with side effects.
                      # DatabaseManager connections might be autocommit=True.
        return csv_output_string
    except Exception as e:
        if conn: # Ensure rollback if connection was established
            try:
                conn.rollback() # Rollback on the specific connection used
            except Exception as rb_e:
                logger.error(f"Error during rollback: {rb_e}")
        error_message = str(e).replace('\n', ' ').strip()
        logger.error(f"Error executing SQL function: {error_message} SQL: {sql_query}")
        return f"Error executing query: {error_message}"
    finally:
        if conn:
            db_manager.release_connection(conn) # Ensure connection is released

execute_sql: BaseTool = tool(execute_sql_func)
execute_sql.name = "SQL_Executor"
