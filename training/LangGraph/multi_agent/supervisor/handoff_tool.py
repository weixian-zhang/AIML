
from langchain.tools import tool, BaseTool
from langgraph.types import Command
from state import AgentState

# https://www.youtube.com/watch?app=desktop&v=p1c_pm6LWI0
# https://thinknew.tech/blog/handoff-between-agents-building-effective-multi-agent-systems-with-langgraph
from typing import Annotated
from langgraph.prebuilt import InjectedState

def create_handoff_tool(next_agent_name: str, description: str, graph='') -> BaseTool:

    @tool(description=description)
    def handoff_tool(state: Annotated[dict, InjectedState]) -> Command:
        messages = state['messages']
        state['web_search_content'] = messages[-1].content if state.messages else ""
        return Command(goto=next_agent_name, update=state, graph=graph)

    return handoff_tool