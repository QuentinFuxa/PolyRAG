import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from langchain_core.tools import BaseTool, tool
from agents._graph_store import GraphStore
from typing import Optional, List, Dict, Any, Union, Callable
from db_manager import DatabaseManager

graph_store = GraphStore()
db_manager = DatabaseManager()

def create_graph(
    query: Optional[str] = None, # SQL query to fetch data (use either query or data)
    data: Optional[List[Dict[str, Any]]] = None, # Direct data input as list of dicts (use either query or data)
    chart_type: str ='bar',      # chart type (bar, scatter, line, pie, etc.)
    x_col: str = None,            # column name for the X-axis
    y_col: str = None,            # column name for the Y-axis
    color_col: str = None,        # column name for color encoding (categories)
    size_col: str = None,         # column name for size encoding (e.g., scatter bubble chart)
    title: str='My Graph',      # chart title
    width: int =800,             # chart width
    height: int=600,            # chart height
    orientation:str='v',       # orientation 'v' or 'h' for bar charts
    labels: dict=None,           # dictionary for renaming axis labels or legend labels
    template: str='plotly_white', # chart style template
    markers: bool = False,        # whether to show markers on line charts
    preprocess: Optional[Dict[str, Union[bool, str, List[str]]]] = None  # preprocessing instructions
):
    """
    Generates a Plotly chart as JSON from either a SQL query or directly provided data.

    The tool returns a graph ID which will be rendered automatically by the frontend.
    DO NOT create links or references to the graph in your response.
    Just acknowledge that a graph has been created and explain what it shows.

    Args:
        query (Optional[str]): A valid SQL query to fetch data. Provide either 'query' or 'data'.
        data (Optional[List[Dict[str, Any]]]): Data provided directly as a list of dictionaries. Provide either 'query' or 'data'.
        chart_type (str): The type of chart to generate (bar, scatter, line, pie, etc.)
        x_col (str): The column name for the X-axis.
        y_col (str): The column name for the Y-axis.
        color_col (str): The column name for color encoding (categories)
        size_col (str): The column name for size encoding (e.g., scatter bubble chart)
        title (str): The chart title
        width (int): The chart width
        height (int): The chart height
        orientation (str): The orientation for bar charts ('v' or 'h')
        labels (dict): Dictionary for renaming axis labels or legend labels
        template (str): The chart style template
        markers (bool): Whether to show markers on line charts (default: False)
        preprocess (Optional[Dict]): Preprocessing instructions for data transformation. Supported options:
            - create_date: Set to 'from_year_month' to create a 'date' column from 'year' and 'month'
            - use_date_for_x: Set to True to use the created 'date' column as x-axis (default: True)
            - aggregate: Dict with settings for data aggregation:
                - group_by: Column to group by
                - column: Column to aggregate
                - function: Aggregation function ('sum', 'mean', etc.)
                - new_column: Name for the aggregated column (default: {column}_{function})
    """

    # print('Graph with the following parameters:')
    # print(f"Query: {query}")
    # print(f"Chart Type: {chart_type}")
    # print(f"X Column: {x_col}")
    # print(f"Y Column: {y_col}")
    # print(f"Color Column: {color_col}")
    # print(f"Size Column: {size_col}")
    # print(f"Title: {title}")
    # print(f"Width: {width}")
    # print(f"Height: {height}")
    # print(f"Orientation: {orientation}")
    # print(f"Labels: {labels}")
    # print(f"Template: {template}")

    # Validate input: Ensure either query or data is provided, but not both
    if query and data:
        raise ValueError("Provide either 'query' or 'data', not both.")
    if not query and not data:
        raise ValueError("Must provide either 'query' or 'data'.")

    # 1) Load data into DataFrame
    if data:
        df = pd.DataFrame(data)
    elif query:
        # Execute the SQL query
        cleaned_query = re.sub(r'(?<!%)%(?!%)', '%%', query)
        try:
            df = pd.read_sql(cleaned_query, con=db_manager.engine)
        except Exception as e:
            # Catch potential SQL execution errors (broadly for now)
            # and prefix the error message for identification by the agent.
            raise ValueError(f"SQL Execution Error: Failed to execute query '{cleaned_query}'. Reason: {e}")
    else:
         # This case should not be reached due to validation above, but added for safety
         raise ValueError("No data source provided (neither query nor data).")

    if 'df' in locals() and df.empty:
        
        if query: # check if the error originated from SQL or just empty direct data
             raise ValueError("SQL Execution Error: The query executed successfully but returned no results.")
        else:
             raise ValueError("Input Data Error: The provided 'data' is empty.")

    # preprocessing options before column validation
    if preprocess:
        # Create date column from year and month columns
        if 'create_date' in preprocess:
            if preprocess['create_date'] == 'from_year_month':
                if 'year' in df.columns and 'month' in df.columns:
                    # Create date column using year and month
                    try:
                        # Ensure year and month are integers first, then strings
                        df['year'] = df['year'].astype(int).astype(str)
                        df['month'] = df['month'].astype(int).astype(str)
                        
                        # Pad month with leading zero if needed
                        df['month'] = df['month'].str.zfill(2)
                        
                        # Create the date string in ISO format (YYYY-MM-DD)
                        df['date'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-01')
                        
                        # If x_col is not specified but we're creating a date, assume it's for x-axis
                        if not x_col and preprocess.get('use_date_for_x', True):
                            x_col = 'date'
                        
                        # Explicitly sort by the new date column for time series plots
                        df = df.sort_values(by='date')

                    except Exception as e:
                        raise ValueError(f"Failed to create date column from year and month: {str(e)}")
                else:
                    raise ValueError("Cannot create date column: Either 'year' or 'month' column is missing")
    
        # Handle additional column transformations or feature engineering
        if 'aggregate' in preprocess:
            # Example: Sum values by a grouping column
            agg_config = preprocess['aggregate']
            if isinstance(agg_config, dict):
                group_by = agg_config.get('group_by', None)
                agg_col = agg_config.get('column', None)
                agg_func = agg_config.get('function', 'sum')
                new_col_name = agg_config.get('new_column', f"{agg_col}_{agg_func}")
                
                if group_by and agg_col:
                    if group_by in df.columns and agg_col in df.columns:
                        try:
                            # Perform the aggregation
                            df_agg = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
                            df_agg.rename(columns={agg_col: new_col_name}, inplace=True)
                            
                            # Replace the DataFrame with the aggregated one
                            df = df_agg
                            
                            # Update column names if they were specified and they correspond to old columns
                            if x_col == group_by:
                                x_col = group_by
                            if y_col == agg_col:
                                y_col = new_col_name
                        except Exception as e:
                            raise ValueError(f"Failed to aggregate data: {str(e)}")
                    else:
                        missing_cols = []
                        if group_by not in df.columns:
                            missing_cols.append(group_by)
                        if agg_col not in df.columns:
                            missing_cols.append(agg_col)
                        raise ValueError(f"Cannot aggregate data: Missing columns: {', '.join(missing_cols)}")
    
    # Verify specified columns exist (after preprocessing, only if df exists and is not empty)
    if 'df' in locals() and not df.empty:
        if x_col and x_col not in df.columns:
            columns_str = ", ".join(df.columns)
            raise ValueError(f"Data Error: Column '{x_col}' for X-axis not found. Available columns: {columns_str}")
        if y_col and y_col not in df.columns:
            columns_str = ", ".join(df.columns)
            raise ValueError(f"Data Error: Column '{y_col}' for Y-axis not found. Available columns: {columns_str}")
        if color_col and color_col not in df.columns:
            columns_str = ", ".join(df.columns)
            raise ValueError(f"Data Error: Column '{color_col}' for color not found. Available columns: {columns_str}")
        if size_col and size_col not in df.columns:
            columns_str = ", ".join(df.columns)
            raise ValueError(f"Data Error: Column '{size_col}' for size not found. Available columns: {columns_str}")
    elif 'df' not in locals() or df.empty:
        # This case should ideally be caught earlier, but as a safeguard:
        # If df doesn't exist or is empty, we can't proceed with plotting.
        # The specific error (SQL vs Data) should have been raised already.
        # If somehow we reach here, raise a generic error.
        raise ValueError("Data Error: Cannot proceed with plotting as data is unavailable or empty.")


    fig = None

    # 2) Generate the Plotly chart based on chart_type
    if chart_type == 'bar':
        # Bar chart
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            orientation=orientation,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height
        )
    elif chart_type == 'scatter':
        # Scatter (point cloud)
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height
        )
    elif chart_type == 'line':
        # Line chart
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height,
            markers=markers
        )
        
        if markers and not hasattr(px.line, 'markers'):
            fig.update_traces(mode='lines+markers')
    elif chart_type == 'pie':
        # Pie chart
        if not x_col:
            raise ValueError("For a pie chart, please specify 'x_col' for category names (or 'values' for quantities).")
        fig = px.pie(
            df,
            names=x_col,
            values=y_col,
            color=color_col,
            title=title,
            template=template,
            width=width,
            height=height
        )
    else:
        # Custom chart type or fallback
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            orientation=orientation,
            labels=labels,
            title=f"Chart type '{chart_type}' not handled, defaulting to bar chart.",
            template=template,
            width=width,
            height=height
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(title=''),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    if x_col == 'date' and preprocess and 'create_date' in preprocess:
         fig.update_layout(
             xaxis_type='date',
             xaxis_tickformat='%b %Y' # Format as 'Month Year', e.g., 'Jan 2022'
         )

    # 4) Export figure to JSON for frontend rendering
    fig_json = fig.to_json()
    id = graph_store.store_graph(fig_json)
    print(f"Graph stored with ID: {id}")
    return id

tool_create_graph: BaseTool = tool(create_graph)
tool_create_graph.name = "Create_Graph"
tool_create_graph.description = """
"""