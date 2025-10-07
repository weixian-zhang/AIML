
from langchain.tools import tool, BaseTool
from langgraph.types import Command
from state import SupervisorState
from langchain_core.messages import ToolMessage

# https://www.youtube.com/watch?app=desktop&v=p1c_pm6LWI0
# https://thinknew.tech/blog/handoff-between-agents-building-effective-multi-agent-systems-with-langgraph
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool, InjectedToolCallId

def create_handoff_tool(next_agent_name: str, description: str, graph='') -> BaseTool:

    @tool(description=description)
    def handoff_tool(state: Annotated[SupervisorState, InjectedState],
                     tool_call_id: Annotated[str, InjectedToolCallId]
                     ) -> Command:
        
        tool_message = ToolMessage(
            content=f"Successfully transferred to {next_agent_name}",
            name=next_agent_name,
            tool_call_id=tool_call_id,
        )

        web_search_content = state.web_search_content

        return Command(goto=next_agent_name, 
                       update={
                           'messages': state.messages + [tool_message],
                           'web_search_content': web_search_content
                       },
                       graph=graph)

    return handoff_tool