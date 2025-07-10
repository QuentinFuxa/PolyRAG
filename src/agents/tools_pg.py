from langchain_core.tools import BaseTool, tool
import os
import csv
import io
from datetime import date, datetime
from decimal import Decimal
from typing import List, Literal, Optional, Union
import logging # Added for logging
from dotenv import load_dotenv

from db_manager import DatabaseManager

logger = logging.getLogger(__name__) # Added logger

load_dotenv()

LANGUAGE = os.environ.get("LANGUAGE", "english")

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
    letter_name_subquery: Optional[str] = None,
    search_text: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Union[tuple[List[dict], int], int]: # For content: (list_of_dicts, total_count), for count: int
    """
    Core function to fetch demand data (content or count) based on optional priority,
    letter filters (names from public.public_data via list or subquery),
    and optional text search within demand content.
    If priority is None, fetches all demands (priority 1 and 2).
    If letter_names/subquery are not provided, operates on all letters.
    """
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

    # Build the base query parts
    if return_type == 'content':
        select_clause = "d.demand_text, l.name, d.priority" # Select demand text, letter name, and priority
    elif return_type == 'count':
        select_clause = "COUNT(*)"
    else:
        raise ValueError("Invalid return_type specified.") # Should not happen if called by tools
    base_query_from_join = """
        FROM public.demands d
        JOIN public.letters l ON d.id_letter = l.id_letter
    """
    
    conditions = []
    params_list = []

    if target_letter_names:
        placeholders = ', '.join(['%s'] * len(target_letter_names))
        conditions.append(f"l.name IN ({placeholders})")
        params_list.extend(target_letter_names)

    # Add priority filter conditionally
    if priority is not None:
        conditions.append("d.priority = %s")
        params_list.append(priority)

    # Add text search filter conditionally
    if search_text:
        # Search on the pre-computed tsvector column in public.demands
        conditions.append(f"d.demand_tsv @@ plainto_tsquery(%s, %s)")
        params_list.extend([LANGUAGE, search_text])

    final_query = f"SELECT {select_clause} {base_query_from_join}"
    if conditions:
        final_query += " WHERE " + " AND ".join(conditions)

    # Add ordering for content retrieval
    if return_type == 'content':
        final_query += " ORDER BY l.name, d.start"
        # Add LIMIT and OFFSET for content queries if provided
        if limit is not None:
            final_query += " LIMIT %s"
            params_list.append(limit)
        if offset is not None:
            final_query += " OFFSET %s"
            params_list.append(offset)

    params_for_main_query = tuple(params_list)
    results = _execute_safe_query(final_query, params_for_main_query)

    # Process results
    if return_type == 'count':
        return results[0][0] if results else 0
    elif return_type == 'content':
        # Construct list of dictionaries
        demands_list = [
            {"text": row[0], "letter_name": row[1], "priority": row[2]}
            for row in results
            if row[0] is not None # Ensure demand_text is not None
        ]
        
        # Get total count for pagination info (without limit/offset)
        count_query_parts = [f"SELECT COUNT(*) {base_query_from_join}"]
        count_params_list = [] # Separate params list for count query
        
        # Re-apply same conditions for count query
        if target_letter_names:
            placeholders = ', '.join(['%s'] * len(target_letter_names))
            count_query_parts.append(f"l.name IN ({placeholders})")
            count_params_list.extend(target_letter_names)
        if priority is not None:
            count_query_parts.append("d.priority = %s")
            count_params_list.append(priority)
        if search_text:
            count_query_parts.append(f"d.demand_tsv @@ plainto_tsquery(%s, %s)")
            count_params_list.extend([LANGUAGE, search_text])

        final_count_query = count_query_parts[0]
        if len(count_query_parts) > 1:
            final_count_query += " WHERE " + " AND ".join(count_query_parts[1:])
        
        total_count_results = _execute_safe_query(final_count_query, tuple(count_params_list))
        total_matching_demands = total_count_results[0][0] if total_count_results else 0
        
        return demands_list, total_matching_demands
    # else case for invalid return_type is handled by initial check


@tool
def get_demand_content(
    demand_type: Optional[Literal["Demandes à traiter prioritairement", "Autres demandes"]] = None, # Made optional
    letter_names: Optional[List[str]] = None,
    letter_name_subquery: Optional[str] = None,
    search_text: Optional[str] = None,
    limit: Optional[int] = 100, # Default limit
    offset: Optional[int] = 0   # Default offset
) -> dict: # Returns a dictionary with demands and pagination info
    """
    Retrieves the text content of demands, along with their source letter and priority.
    Optionally filters by demand_type, letter_names (or letter_name_subquery), and search_text.
    If letter_names/subquery are omitted, retrieves demands from all letters.
    If demand_type is not specified, retrieves all demands (priority 1 and 2).
    Can also filter demands by searching for specific text within their content using full-text search.

    Args:
        demand_type: (Optional) The type of demand ('Demandes à traiter prioritairement' or 'Autres demandes'). Defaults to all.
        letter_names: (Optional) A list of exact letter names (strings) for the letters to search within (filters on `public.letters.name`). Use this OR letter_name_subquery.
        letter_name_subquery: (Optional) A SQL SELECT query string targeting `public.public_data` that returns a list of letter names (filters on `public.letters.name`).
                              Example: "SELECT name FROM public.public_data WHERE site_name = 'Blayais'". Use this OR letter_names.
        search_text: (Optional) Text to search for within the pre-extracted demand content (in `public.demands.demand_tsv`)
                     using advanced full-text search.
                     (language for query parsing configured via LANGUAGE env var, defaults to English).
                     If provided, only demands matching this text will be returned.
        limit: (Optional) Maximum number of demands to return. Defaults to 100.
        offset: (Optional) Number of demands to skip before returning results. Defaults to 0.

    Returns:
        A dictionary containing:
            - "demands": A list of dictionaries, each with "text", "letter_name", and "priority".
            - "total_matching_demands": The total number of demands matching the criteria (ignoring limit/offset).
            - "limit": The limit used for the query.
            - "offset": The offset used for the query.
            - "has_more": Boolean indicating if more results are available beyond the current set.
        Returns an error dictionary on failure (e.g., {"error": "message"}).
    """
    try:
        priority_val = None
        if demand_type:
            priority_map = {
                "Demandes à traiter prioritairement": 1,
                "Autres demandes": 2
            }
            priority_val = priority_map.get(demand_type)
            if priority_val is None and demand_type is not None:
                return {"error": "Invalid demand_type specified."}

        # _get_demands_data for 'content' now returns a tuple: (demands_list, total_count)
        demands_list, total_matching_demands = _get_demands_data(
            return_type='content',
            priority=priority_val,
            letter_names=letter_names,
            letter_name_subquery=letter_name_subquery,
            search_text=search_text,
            limit=limit,
            offset=offset
        )
        
        return {
            "demands": demands_list,
            "total_matching_demands": total_matching_demands,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(demands_list)) < total_matching_demands
        }
    except (ValueError, ConnectionError, Exception) as e:
        return {"error": str(e)}


@tool
def count_demands(
    demand_type: Optional[Literal["Demandes à traiter prioritairement", "Autres demandes"]] = None, # Made optional
    letter_names: Optional[List[str]] = None,
    letter_name_subquery: Optional[str] = None,
    search_text: Optional[str] = None
) -> Union[int, str]:
    """
    Counts the number of demands.
    Optionally filters by demand_type, letter_names (or letter_name_subquery), and search_text.
    If letter_names/subquery are omitted, counts demands from all letters.
    If demand_type is not specified, counts all demands (priority 1 and 2).
    Can also filter demands by searching for specific text within their content using full-text search.

    Args:
        demand_type: (Optional) The type of demand to count ('Demandes à traiter prioritairement' or 'Autres demandes'). Defaults to all.
        letter_names: (Optional) A list of exact letter names (strings) for the letters to search within (filters on `public.letters.name`). Use this OR letter_name_subquery.
        letter_name_subquery: (Optional) A SQL SELECT query string targeting `public.public_data` that returns a list of letter names (filters on `public.letters.name`).
                              Example: "SELECT name FROM public.public_data WHERE site_name = 'Blayais'". Use this OR letter_names.
        search_text: (Optional) Text to search for within the pre-extracted demand content (in `public.demands.demand_tsv`)
                     using advanced full-text search.
                     (language for query parsing configured via LANGUAGE env var, defaults to English).
                     If provided, only demands matching this text will be counted.

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
            if priority is None and demand_type is not None: # Check if demand_type was given but not mapped
                 # This case should ideally not happen due to Literal typing, but good practice
                return "Error: Invalid demand_type specified."
        # Always call _get_demands_data, priority will be None if demand_type was not specified
        return _get_demands_data(
            return_type='count',
            priority=priority, # Pass None if demand_type wasn't specified or mapped
            letter_names=letter_names,
            letter_name_subquery=letter_name_subquery,
            search_text=search_text
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
