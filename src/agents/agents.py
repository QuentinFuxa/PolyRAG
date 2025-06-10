from dataclasses import dataclass

from langgraph.pregel import Pregel
from src.schema import AgentInfo

try:
    from agents.pg_rag_assistant import pg_rag_assistant
except ImportError:
    from pg_rag_assistant import pg_rag_assistant

DEFAULT_AGENT = "pg_rag_assistant"


@dataclass
class Agent:
    description: str
    graph: Pregel


agents: dict[str, Agent] = {
    "pg_rag_assistant": Agent(
        description="An assistant that has access to a postgres database, plotly and pdf visualisation tools.", graph=pg_rag_assistant
    )
}


def get_agent(agent_id: str) -> Pregel:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
