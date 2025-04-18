import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from langchain_core.tools import BaseTool, tool
from agents._graph_store import GraphStore
from db_manager import DatabaseManager



graph_store = GraphStore()
db_manager = DatabaseManager()

def create_graph(
    query: str,            # SQL query
    chart_type: str ='bar',      # chart type (bar, scatter, line, pie, etc.)
    x_col: str = None,            # column name for the X-axis
    y_col: str = None,            # column name for the Y-axis
    color_col: str = None,        # column name for color encoding (categories)
    size_col: int =None,         # column name for size encoding (e.g., scatter bubble chart)
    title: str='My Graph',      # chart title
    width: int =800,             # chart width
    height: int=600,            # chart height
    orientation:str='v',       # orientation 'v' or 'h' for bar charts
    labels: dict=None,           # dictionary for renaming axis labels or legend labels
    template: str='plotly_white' # chart style template
):
    """
    Executes the SQL query, stores the result in a Pandas DataFrame, 
    and generates a Plotly chart as JSON.
    
    The tool returns graph which will be rendered automatically - 
    DO NOT create links or references to the graph in your response.
    Just acknowledge that a graph has been created and explain what it shows.
    
    Args:
        query (str): A valid SQL query
        chart_type (str): The type of chart to generate (bar, scatter, line, pie, etc.)
        x_col (str): The column name for the X-axis
        y_col (str): The column name for the Y-axis
        color_col (str): The column name for color encoding (categories)
        size_col (str): The column name for size encoding (e.g., scatter bubble chart)
        title (str): The chart title
        width (int): The chart width
        height (int): The chart height
        orientation (str): The orientation for bar charts ('v' or 'h')
        labels (dict): Dictionary for renaming axis labels or legend labels
        template (str): The chart style template
    """

    print('Graph with the following parameters:')
    print(f"Query: {query}")
    print(f"Chart Type: {chart_type}")
    print(f"X Column: {x_col}")
    print(f"Y Column: {y_col}")
    print(f"Color Column: {color_col}")
    print(f"Size Column: {size_col}")
    print(f"Title: {title}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Orientation: {orientation}")
    print(f"Labels: {labels}")
    print(f"Template: {template}")
    
    # 1) Execute the SQL query
    df = pd.read_sql(query, con=db_manager.engine)

    # Verify we have a non-empty DataFrame
    if df.empty:
        raise ValueError("The SQL query returned no results or the DataFrame is empty.")

    # Verify specified columns exist
    if x_col and x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in the DataFrame.")
    if y_col and y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in the DataFrame.")
    if color_col and color_col not in df.columns:
        raise ValueError(f"Column '{color_col}' does not exist in the DataFrame.")
    if size_col and size_col not in df.columns:
        raise ValueError(f"Column '{size_col}' does not exist in the DataFrame.")

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
            height=height
        )
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

    # 3) Extra customization via fig.update_layout or fig.update_traces if needed
    fig.update_layout(
        showlegend=True,
        legend=dict(title=''),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # 4) Export figure to JSON for frontend rendering
    fig_json = fig.to_json()
    id = graph_store.store_graph(fig_json)
    print(f"Graph stored with ID: {id}")
    return id

display_graph: BaseTool = tool(create_graph)
display_graph.name = "Graph_Viewer"