from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.tools_plotly import create_graph # Tool is named Create_Graph
from core import get_model, settings

# Define the state for the GraphingAgent
class GraphingAgentState(MessagesState, total=False):
    """
    input_data: The structured data (list of dictionaries) for graphing. Can be None if a query is to be used.
    query_string: An optional SQL query string to fetch data for the graph.
    graph_instructions: Natural language instructions for the graph.
    language: The language for graph elements like title, axis labels, legend. Defaults to "english".
    graph_id: The ID of the generated graph, returned by Create_Graph.
    error: Any error message encountered during graph generation.
    """
    input_data: Optional[List[Dict[str, Any]]]
    query_string: Optional[str]
    graph_instructions: str
    language: str = "english"
    graph_id: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 1

GRAPHING_AGENT_SYSTEM_PROMPT = """
You are a specialized data visualization assistant.
Your sole purpose is to create Plotly graphs using the 'Create_Graph' tool based on provided data and instructions.

Your key capabilities:
- Creating various chart types (bar, line, scatter, pie)
- Working with direct data input or SQL queries
- Translating chart elements to different languages
- Preprocessing data automatically (creating date columns, aggregating values)
- Handling common errors internally without returning them to the user

You will receive:
1. `graph_instructions`: A natural language description of the desired graph
2. `language`: Target language for graph elements (default: "english")
3. `input_data` (optional): A JSON list of dictionaries containing the data to plot
4. `query_string` (optional): A SQL query string to fetch data

## Smart Data Preprocessing

The Create_Graph tool now has advanced preprocessing capabilities. Instead of failing when columns don't exist, you can instruct it to create them:

1. For data with separate `year` and `month` columns:
   - Set `preprocess={'create_date': 'from_year_month', 'use_date_for_x': True}` 
   - This automatically creates a `date` column and uses it for the x-axis

2. For data needing aggregation:
   - Set `preprocess={'aggregate': {'group_by': 'category', 'column': 'sales', 'function': 'sum', 'new_column': 'total_sales'}}`
   - This groups and aggregates data before plotting

## Required Parameters for ALL Charts

- `chart_type`: Chart type ('bar', 'scatter', 'line', 'pie')
- `title`: Chart title (derived from instructions and translated if needed)
- `labels`: Dictionary for renaming columns to more readable labels (e.g. `{'original_col_name': 'Readable Label'}`). Ensure all data-mapped columns (`x_col`, `y_col`, `color_col`, `size_col`, `symbol_col`, `facet_row`, `facet_col`, `line_group`, `line_dash`) have corresponding entries if you want to rename them.
- One of:
  - `data`: Use this when `input_data` is provided
  - `query`: Use this when `query_string` is provided, or when you need to formulate a SQL query

## Common Chart Parameters (Use as applicable)

- `x_col`, `y_col`: Specify correct column names for X and Y axes.
- `color_col`: Column for color encoding (grouping data by color).
- `size_col`: Column for size encoding (e.g., bubble size in scatter plots).
- `symbol_col`: Column for symbol encoding (different marker shapes in scatter/line plots).
- `hover_data`: List of column names to add to hover tooltips for more detail.
- `text_auto`: Set to `True` to display values on bars/markers, or a d3-format string (e.g., `'.2s'`) for custom formatting.
- `facet_row`, `facet_col`: Columns to create a grid of subplots (facets).
- `log_x`, `log_y`: Set to `True` to use a logarithmic scale for the X or Y axis, respectively.

## Chart-Specific Parameters

- **Bar Charts:**
    - `barmode`: Controls how bars are displayed when multiple categories exist for the same x-value.
        - `'group'`: Bars are placed side-by-side.
        - `'stack'`: Bars are stacked on top of each other (default if `color_col` is used).
        - `'relative'`: Similar to stack, but negative values are stacked below the axis.
    - `orientation`: `'v'` for vertical (default), `'h'` for horizontal.
- **Line Charts:**
    - `markers`: Set to `True` to show markers on data points along the lines. (Strongly Recommended)
    - `line_group`: Column to group data for drawing separate lines, useful if `color_col` is used for another purpose or if you want lines not distinguished by color.
    - `line_dash`: Column to map to different dash styles for lines (e.g., solid, dashed, dotted).
- **Scatter Plots:**
    - `trendline`: Adds a trendline. Use `'ols'` for an Ordinary Least Squares regression line.
- **Pie Charts:**
    - `names` (maps to `x_col`): Column for category names of pie slices.
    - `values` (maps to `y_col`): Column for the numerical values determining slice sizes.

## Translation

- If `language` is not "english", translate:
  - Chart `title`
  - All values in the `labels` dictionary

## Error Prevention

Before calling the Create_Graph tool, you MUST:
1. Check if needed columns exist and use preprocessing if they don't
2. For time-based data with year and month columns, automatically use preprocessing
3. Provide helpful column names in error messages

## Success Confirmation

After a successful Create_Graph call (when you receive a graph_id), simply reply: "Graph created with ID: {graph_id}"

## Examples:

Example 1: Bar chart with English labels
```
Input:
- graph_instructions: "Bar chart of total sales per product category. Title it 'Product Sales Performance'."
- language: "english"
- input_data: [{'category': 'Electronics', 'total_sales': 15000}, {'category': 'Books', 'total_sales': 8000}]

Output:
Create_Graph(
  data=input_data,
  chart_type='bar',
  x_col='category',
  y_col='total_sales',
  title='Product Sales Performance',
  labels={'category': 'Product Category', 'total_sales': 'Total Sales'}
)
```

Example 2: Line chart with preprocessing (date creation) and French translation
```
Input:
- graph_instructions: "Comparative evolution of inspections by month between sites since 2022"
- language: "french"
- input_data: [
    {"site_name":"Tricastin","year":2022,"month":1,"inspection_count":5},
    {"site_name":"Tricastin","year":2022,"month":2,"inspection_count":8},
    {"site_name":"Gravelines","year":2022,"month":1,"inspection_count":4}
  ]

Output:
Create_Graph(
  data=input_data,
  chart_type='line',
  preprocess={'create_date': 'from_year_month', 'use_date_for_x': True},
  x_col='date',
  y_col='inspection_count',
  color_col='site_name',
  markers=True,
  title='Évolution comparative du nombre d\'inspections par mois depuis 2022',
  labels={
    'date': 'Date',
    'inspection_count': 'Nombre d\'inspections',
    'site_name': 'Site'
  }
)
```

Example 3: Pie chart using a SQL query
```
Input:
- graph_instructions: "Pie chart of user distribution by country from the 'users' table"
- language: "english"
- query_string: "SELECT country_code, COUNT(user_id) AS user_count FROM users GROUP BY country_code"

Output:
Create_Graph(
  query=query_string,
  chart_type='pie',
  x_col='country_code',
  y_col='user_count',
  title='User Distribution by Country',
  labels={'country_code': 'Country', 'user_count': 'Number of Users'}
)
```

Example 4: Automatic data aggregation using preprocessing
```
Input:
- graph_instructions: "Bar chart of total sales by category"
- language: "english"
- input_data: [
    {"category":"Electronics","product":"Phone","sales":500},
    {"category":"Electronics","product":"Laptop","sales":1200},
    {"category":"Books","product":"Fiction","sales":300}
  ]

Output:
Create_Graph(
  data=input_data,
  chart_type='bar',
  preprocess={
    'aggregate': {
      'group_by': 'category',
      'column': 'sales',
      'function': 'sum',
      'new_column': 'total_sales'
    }
  },
  x_col='category',
  y_col='total_sales',
  title='Total Sales by Category',
  labels={'category': 'Category', 'total_sales': 'Total Sales'}
)
```

Example 5: Line chart using SQL query with year/month extraction and French translation
```
Input:
- graph_instructions: "Comparative evolution of the number of inspections per month between Tricastin and Gravelines since 2022."
- language: "french"
- query_string: "SELECT site_name, EXTRACT(YEAR FROM sent_date) AS year, EXTRACT(MONTH FROM sent_date) AS month, COUNT(*) AS inspection_count FROM public.public_data WHERE site_name IN ('Tricastin', 'Gravelines') AND sent_date >= '2022-01-01' GROUP BY site_name, year, month ORDER BY site_name, year, month;"

Output:
# Note: Even though the query returns 'year' and 'month', we use preprocessing to create a 'date' column for plotting.
Create_Graph(
  query=query_string, # Use the provided SQL query
  chart_type='line',
  preprocess={'create_date': 'from_year_month', 'use_date_for_x': True}, # Create 'date' from 'year'/'month'
  x_col='date', # Use the generated 'date' column for the x-axis
  y_col='inspection_count',
  color_col='site_name',
  markers=True,
  title='Évolution comparative du nombre d\'inspections par mois depuis 2022',
  labels={
    'date': 'Date',
    'inspection_count': 'Nombre d\'inspections',
    'site_name': 'Site'
  }
)

Example 6: Bar plot of comparison on several months:
Input :{"graph_instructions":"Créer un graphique comparant le nombre de demandes prioritaires et non prioritaires entre janvier et mars 2025. Les données sont : Janvier - 24 prioritaires, 416 non prioritaires ; Février - 27 prioritaires, 639 non prioritaires ; Mars - 0 prioritaires, 18 non prioritaires.","language":"french","input_data":[{"mois":"Janvier","prioritaires":24,"non_prioritaires":416},{"mois":"Février","prioritaires":27,"non_prioritaires":639},{"mois":"Mars","prioritaires":0,"non_prioritaires":18}]}

Output:
Create_Graph(
  data=[{"mois": "Janvier", "type": "prioritaires", "nombre": 24},
    {"mois": "Janvier", "type": "non_prioritaires", "nombre": 416},
    {"mois": "Février", "type": "prioritaires", "nombre": 27},
    {"mois": "Février", "type": "non_prioritaires", "nombre": 639},
    {"mois": "Mars", "type": "prioritaires", "nombre": 0},
    {"mois": "Mars", "type": "non_prioritaires", "nombre": 18}],
  chart_type="bar",
  x_col="mois",
  y_col="nombre",
  color_col="type",
  title="Comparaison des demandes prioritaires et non prioritaires (Janv–Mars 2025)",
  labels={
    "mois": "Mois",
    "nombre": "Nombre de demandes",
    "type": "Type de demande"
  },
  barmode="group" # Added barmode for clarity, assuming data is pre-shaped for grouping.
)
```

Example 7: Scatter plot with symbols, hover data, and trendline
```
Input:
- graph_instructions: "Scatter plot of engine displacement vs. fuel efficiency. Color by car make, use different symbols for transmission type, and show horsepower on hover. Add a trendline. Title: 'Fuel Efficiency Analysis'."
- language: "english"
- input_data: [
    {"make": "Toyota", "displacement": 1.8, "efficiency": 30, "transmission": "Auto", "horsepower": 130},
    {"make": "Honda", "displacement": 1.5, "efficiency": 35, "transmission": "Manual", "horsepower": 120},
    {"make": "Ford", "displacement": 2.5, "efficiency": 22, "transmission": "Auto", "horsepower": 170}
  ]

Output:
Create_Graph(
  data=input_data,
  chart_type='scatter',
  x_col='displacement',
  y_col='efficiency',
  color_col='make',
  symbol_col='transmission',
  hover_data=['horsepower'],
  trendline='ols',
  title='Fuel Efficiency Analysis',
  labels={
    'displacement': 'Engine Displacement (L)',
    'efficiency': 'Fuel Efficiency (MPG)',
    'make': 'Car Make',
    'transmission': 'Transmission Type',
    'horsepower': 'Horsepower'
  }
)
```

Example 8: Faceted bar chart with text values displayed
```
Input:
- graph_instructions: "Bar chart of average scores by subject, faceted by school. Display the average scores on the bars. Title: 'Average Scores by Subject and School'."
- language: "english"
- input_data: [
    {"school": "North High", "subject": "Math", "avg_score": 85},
    {"school": "North High", "subject": "Science", "avg_score": 90},
    {"school": "South High", "subject": "Math", "avg_score": 82},
    {"school": "South High", "subject": "Science", "avg_score": 88}
  ]

Output:
Create_Graph(
  data=input_data,
  chart_type='bar',
  x_col='subject',
  y_col='avg_score',
  facet_col='school',
  text_auto=True,
  title='Average Scores by Subject and School',
  labels={
    'subject': 'Subject',
    'avg_score': 'Average Score',
    'school': 'School'
  }
)
```

Example 9: Line chart with different dash styles and log scale
```
Input:
- graph_instructions: "Line chart showing sensor readings over time for different sensor types. Use dashed lines for 'beta' sensors. Y-axis should be logarithmic. Title: 'Sensor Readings (Log Scale)'."
- language: "french"
- input_data: [
    {"time": 1, "reading": 10, "sensor_type": "alpha"}, {"time": 2, "reading": 100, "sensor_type": "alpha"},
    {"time": 1, "reading": 5, "sensor_type": "beta"}, {"time": 2, "reading": 50, "sensor_type": "beta"}
  ]

Output:
Create_Graph(
  data=input_data,
  chart_type='line',
  x_col='time',
  y_col='reading',
  color_col='sensor_type', # Could also use line_group if color is for something else
  line_dash='sensor_type', # This will make 'beta' different if mapped in Plotly's dash sequences
  markers=True,
  log_y=True,
  title='Relevés des capteurs (échelle logarithmique)',
  labels={
    'time': 'Temps',
    'reading': 'Relevé',
    'sensor_type': 'Type de capteur'
  }
)
```
"""

tools = [create_graph]

async def call_graphing_model(state: GraphingAgentState) -> GraphingAgentState:
    model = get_model(settings.DEFAULT_MODEL).bind_tools(tools, tool_choice="auto")
    messages: List[BaseMessage] = [SystemMessage(content=GRAPHING_AGENT_SYSTEM_PROMPT)]
    
    last_message = state["messages"][-1] if state["messages"] else None
    is_retry = False
    error_message_for_retry = None

    if isinstance(last_message, ToolMessage) and last_message.content.startswith("Error:"):
        is_retry = True
        error_message_for_retry = last_message.content

    if is_retry:
        messages.append(
            HumanMessage(
                content=f"The previous attempt to create the graph failed with the following error:\n"
                        f"'{error_message_for_retry}'\n\n"
                        f"Please analyze the error and try calling the 'Create_Graph' tool again with corrected parameters. "
                        f"Focus on fixing issues related to data selection (columns, types) or chart parameters. "
                        f"Do NOT retry if the error indicates a problem with the SQL query itself (e.g., 'SQL Execution Error'). "
                        f"Original instructions were: {state['graph_instructions']}"
            )
        )
    else:
        if state.get("graph_id"):
             messages.append(
                HumanMessage(
                    content=f"The Create_Graph tool has just successfully returned a graph_id: '{state['graph_id']}'. "
                            "Your ONLY task now is to output a brief confirmation message stating 'Graph created with ID: {state['graph_id']}' and then STOP. "
                            "Do not call any tools. Do not add any other commentary."
                )
            )
        elif state.get("graph_instructions"):
            instruction_message_parts = [
                f"Instructions: {state['graph_instructions']}",
                f"Language for graph elements (titles, axes, legend): {state.get('language', 'english')}"
            ]
            input_data = state.get("input_data")
            query_str = state.get("query_string")

            if input_data is not None:
                instruction_message_parts.append(
                    f"Data Source: Direct `input_data` is available (a list of {len(input_data)} dictionaries)."
                    f" The first row is: {input_data[0] if input_data else 'empty'}."
                    f" Columns available: {list(input_data[0].keys()) if input_data and input_data[0] else 'N/A'}."
                    " You should use the `data` parameter of the 'Create_Graph' tool."
                )
            elif query_str:
                instruction_message_parts.append(
                    f"Data Source: A SQL `query_string` has been provided: \"{query_str}\"."
                    " You should use this with the `query` parameter of the 'Create_Graph' tool."
                )
            else:
                instruction_message_parts.append(
                    "Data Source: Neither `input_data` nor `query_string` was provided. "
                    "If instructions imply data fetching (e.g., from a database), formulate a SQL query and use the `query` parameter of the 'Create_Graph' tool."
                )
            instruction_message = "\n".join(instruction_message_parts)
            messages.append(HumanMessage(content=instruction_message))
        else:
            messages.append(HumanMessage(content="Error: Missing graph instructions in the state."))

    ai_response: AIMessage = await model.ainvoke(messages)
    return {"messages": [ai_response]}

graphing_tool_node = ToolNode(tools)

def handle_tool_result(state: GraphingAgentState) -> GraphingAgentState:
    """Checks the result of the tool call and updates state."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        return {"error": "Unexpected state: Last message was not a ToolMessage."}

    tool_output = last_message.content
    
    if tool_output.startswith("Error:") or tool_output.startswith("SQL Execution Error:") or tool_output.startswith("Data Error:") or tool_output.startswith("Input Data Error:") or tool_output.startswith("ValueError:"):
        is_sql_error = "SQL Execution Error:" in tool_output
        current_retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 1) # Use get for max_retries too for safety
        retries_exhausted = current_retry_count >= max_retries

        if is_sql_error or retries_exhausted:
            # Don't retry SQL errors or if max retries reached
            print(f"Graphing Agent: {'SQL Error' if is_sql_error else 'Max retries reached'}. Stopping. Error: {tool_output}")
            return {"error": tool_output} # Store final error and signal end
        else:
            # Increment retry count for non-SQL errors
            print(f"Graphing Agent: Tool failed (Attempt {current_retry_count + 1}). Retrying. Error: {tool_output}")
            return {"retry_count": current_retry_count + 1, "error": tool_output} # Update retry count, keep error for next model call
    else:
        # Success - tool returned a graph_id
        print(f"Graphing Agent: Tool succeeded. Graph ID: {tool_output}")
        return {"graph_id": tool_output, "error": None, "retry_count": 0} # Store graph_id, clear error, reset retries

graph_builder = StateGraph(GraphingAgentState)

graph_builder.add_node("model", call_graphing_model)
graph_builder.add_node("tools", graphing_tool_node)
graph_builder.add_node("handle_tool_result", handle_tool_result)

graph_builder.set_entry_point("model")

def route_from_model(state: GraphingAgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

def route_from_tool_result(state: GraphingAgentState) -> str:
    error = state.get("error")
    current_retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1) # Use get here as well

    if error and "SQL Execution Error:" not in error and current_retry_count <= max_retries:
        if state.get("graph_id"):
             return "model"
        elif error and ("SQL Execution Error:" in error or current_retry_count > max_retries):
             return END
        elif error:
             return "model"
        else:
             return END

    elif state.get("graph_id"):
        # If we have a graph_id (success), go back to model for confirmation message
        return "model"
    else:
        # If it's an SQL error, or retries are exhausted, or some other terminal state
        return END

graph_builder.add_conditional_edges("model", route_from_model, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "handle_tool_result")
graph_builder.add_conditional_edges("handle_tool_result", route_from_tool_result, {"model": "model", END: END})


# Compile the graph
graphing_agent_graph = graph_builder.compile()


async def run_graphing_agent(
    graph_instructions: str,
    input_data: Optional[List[Dict[str, Any]]] = None,
    query_string: Optional[str] = None,
    language: str = "english"
) -> GraphingAgentState:
    initial_state_args = {
        "graph_instructions": graph_instructions,
        "language": language,
        "messages": []
    }
    if input_data is not None:
        initial_state_args["input_data"] = input_data
    if query_string is not None:
        initial_state_args["query_string"] = query_string
    
    initial_state = GraphingAgentState(**initial_state_args)
    final_state_dict = await graphing_agent_graph.ainvoke(initial_state)

    # The final state dict already contains 'graph_id' or 'error' set by handle_tool_result or the final model call
    # We just need to ensure the type hint matches
    final_state = GraphingAgentState(**final_state_dict)
    
    # Extract the final confirmation message if available
    final_ai_message = None
    for message in reversed(final_state.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
             final_ai_message = message.content
             break # Found the last non-tool-calling AI message

    # Add the final AI message to the state for clarity, if needed elsewhere
    # final_state["final_message"] = final_ai_message

    return final_state

if __name__ == '__main__':
    import asyncio

    async def main_test():
        sample_data = [
            {"year": 2020, "sales": 100, "region": "North"},
            {"year": 2021, "sales": 150, "region": "North"},
            {"year": 2020, "sales": 80, "region": "South"},
            {"year": 2021, "sales": 120, "region": "South"},
        ]
        instructions = "Create a bar chart showing sales per year, colored by region. Title it 'Annual Sales by Region'."
        instructions_french = "Créer un diagramme à barres montrant les ventes par année, coloré par région. Titre : 'Ventes Annuelles par Région'."
        instructions_sql = "Show active user count by month from 'user_activity' table. Title: 'Monthly Active Users'."
        test_query_string = "SELECT strftime('%Y-%m', event_date) AS month, COUNT(DISTINCT user_id) AS active_users FROM user_activity GROUP BY month ORDER BY month;"

        print(f"--- Running graphing agent with instructions (English, with data): {instructions} ---")
        result_state_en_data = await run_graphing_agent(input_data=sample_data, graph_instructions=instructions, language="english")
        print_summary(result_state_en_data)

        print(f"\n--- Running graphing agent with instructions (French, with data): {instructions_french} ---")
        result_state_fr_data = await run_graphing_agent(input_data=sample_data, graph_instructions=instructions_french, language="french")
        print_summary(result_state_fr_data)

        print(f"\n--- Running graphing agent with SQL instructions (English, with query_string): {instructions_sql} ---")
        result_state_sql = await run_graphing_agent(query_string=test_query_string, graph_instructions=instructions_sql, language="english")
        print_summary(result_state_sql)
        
        print(f"\n--- Running graphing agent with instructions only (English, to test LLM query generation): {instructions_sql} ---")
        result_state_llm_query = await run_graphing_agent(input_data=None, query_string=None, graph_instructions=instructions_sql, language="english")
        print_summary(result_state_llm_query)

    def print_summary(result_state: GraphingAgentState):
        print("\nFinal agent state:")
        input_data_val = result_state.get('input_data')
        query_str_val = result_state.get('query_string')

        if input_data_val is not None:
            first_row_summary_list = input_data_val[:1]
            first_row_summary = first_row_summary_list[0] if first_row_summary_list else 'empty'
            num_rows = len(input_data_val)
            print(f"  Input Data Provided: Yes ({first_row_summary}... {num_rows} rows)")
        else:
            print("  Input Data Provided: No")

        if query_str_val is not None:
            print(f"  Query String Provided: \"{query_str_val}\"")
        else:
            print("  Query String Provided: No")
        
        print(f"  Instructions: {result_state.get('graph_instructions', 'N/A')}")
        print(f"  Language: {result_state.get('language', 'N/A')}")
        print(f"  Graph ID: {result_state.get('graph_id', 'N/A')}")
        print(f"  Error: {result_state.get('error', 'N/A')}")
        print("  Messages:")
        for msg in result_state.get("messages", []):
            msg_content = msg.content if msg.content else ''
            print(f"    - {msg.type}: {msg_content[:100]}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"      Tool Calls: {msg.tool_calls}")

    asyncio.run(main_test()) # Comment out for production
