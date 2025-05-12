from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

from agents.graphing_agent import run_graphing_agent, GraphingAgentState


@tool
async def tool_graphing_agent(
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
    # Input validation
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

        # Check for graph_id (successful graph creation)
        if result_state.get("graph_id"):
            return result_state["graph_id"]

        # Check for error message
        elif result_state.get("error"):
            return f"GraphingAgent Error: {result_state['error']}"
        
        # Check messages for final confirmation or error from GraphingAgent's LLM
        final_messages = result_state.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]
            
            # Check if it's an AIMessage with content
            if last_message.type == "ai" and not hasattr(last_message, 'tool_calls') and last_message.content:
                return f"GraphingAgent finished with message: \"{last_message.content}\""
                
            # Check if it's a ToolMessage (output from Graph_Viewer)
            elif last_message.type == "tool" and last_message.content:
                return f"GraphingAgent tool execution result: \"{last_message.content}\""

        # Fallback for unhandled cases
        return "GraphingAgent finished but no clear result was found."

    except Exception as e:
        return f"Error invoking GraphingAgent: {str(e)}"
