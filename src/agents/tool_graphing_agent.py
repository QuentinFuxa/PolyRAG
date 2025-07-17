from typing import Optional, List, Dict, Any
from agents.graphing_agent import run_graphing_agent, GraphingAgentState
from langchain_core.tools import BaseTool, tool


async def graphing_agent(
    graph_instructions: str,
    language: str = "english",
    input_data: Optional[List[Dict[str, Any]]] = None,
    query_string: Optional[str] = None
) -> str:
    """
    Invokes the specialized GraphingAgent to create a visualization.
    You MUST provide graph_instructions (natural language for title, chart type, etc.).
    You MUST provide EITHER input_data (list of dictionaries) OR a query_string (SQL), but not both.
    The language parameter (e.g., 'english', 'french') determines the language of titles and labels.

    Args:
        graph_instructions: Natural language description of the desired graph.
        language: Target language for graph elements (default: "english").
        input_data: Optional. A list of dictionaries containing the data to plot.
        query_string: Optional. A SQL query string to fetch data for the graph.

    Returns:
        A string confirming success and including the graph_id, or an error message.
    """
    if input_data and query_string:
        return "Error: Provide either 'input_data' or 'query_string' to GraphingAgent, not both."
    if not input_data and not query_string:
        return "Error: GraphingAgent requires either 'input_data' or 'query_string'."

    try:
        result_state: GraphingAgentState = await run_graphing_agent(
            graph_instructions=graph_instructions,
            input_data=input_data,
            query_string=query_string,
            language=language
        )
        if result_state.get("graph_id"):
            return result_state["graph_id"]
        elif result_state.get("error"):
            return f"GraphingAgent Error: {result_state['error']}"
        
        final_messages = result_state.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]
            
            if last_message.type == "ai" and not hasattr(last_message, 'tool_calls') and last_message.content:
                return f"GraphingAgent finished with message: \"{last_message.content}\""                
            elif last_message.type == "tool" and last_message.content:
                return f"GraphingAgent tool execution result: \"{last_message.content}\""
        return "GraphingAgent finished but no clear result was found."
    except Exception as e:
        return f"Error invoking GraphingAgent: {str(e)}"

tool_graphing_agent: BaseTool = tool(graphing_agent)
tool_graphing_agent.name = "Graphing_Agent"
tool_graphing_agent.description = """
Generates Plotly graphs from structured tabular data or SQL queries by delegating the heavy-lifting
to a specialised multi-step `GraphingAgent`.

Required arguments
------------------
graph_instructions (str)
    Natural-language description of the desired visualisation: chart type, title, axes, colours,
    grouping logic, etc.

Exactly **one** of the two data-source arguments must be supplied
---------------------------------------------------------------
input_data (list[dict])
    A list of dictionaries representing the rows to plot.
query_string (str)
    A SQL query that returns the data to plot.

Optional arguments
------------------
language (str, default "english")
    Language in which titles, axis labels and legend entries will be generated.

Return value
------------
str
    On success – the unique ``graph_id`` that the frontend can render.
    On failure – an error string starting with ``Error:`` describing the problem.

Example invocation payload
--------------------------
{
  "graph_instructions": "Create a clustered bar chart to display the transition of regional lung \
tier scores over 5 years based on GOLD status in COPD patients. Use distinct colours for each \
transition type and group the bars by GOLD status for each year. Title the chart \
'Transition of Regional Lung Tier Scores by GOLD Status in COPD Patients (2020 vs 2025)'.",
  "language": "english",
  "input_data": [
    {"Year": "2020", "GOLD_Status": "No COPD",   "Tier_0_to_1": 9.3,  "Tier_1_to_0": 24.9, "Tier_1_to_2": 11.8},
    {"Year": "2020", "GOLD_Status": "GOLD 1-2",  "Tier_0_to_1": 23,   "Tier_1_to_0": 17.6, "Tier_1_to_2": 19.1},
    {"Year": "2020", "GOLD_Status": "GOLD 3-4",  "Tier_1_to_2": 23.1, "Tier_2_to_3": 22.3},
    {"Year": "2025", "GOLD_Status": "No COPD",   "Tier_0_to_1": 9.3,  "Tier_1_to_0": 24.9, "Tier_1_to_2": 11.8},
    {"Year": "2025", "GOLD_Status": "GOLD 1-2",  "Tier_0_to_1": 23,   "Tier_1_to_0": 17.6, "Tier_1_to_2": 19.1},
    {"Year": "2025", "GOLD_Status": "GOLD 3-4",  "Tier_1_to_2": 23.1, "Tier_2_to_3": 22.3}
  ]
}
"""
