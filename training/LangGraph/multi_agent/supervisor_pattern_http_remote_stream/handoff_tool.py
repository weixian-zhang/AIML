
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

    '''
    - Creates a handoff tool for delegating tasks between agents.
    - state is injected and is immutable, so we need to return updated state in Command
    '''

    @tool(description=description)
    def handoff_tool(state: Annotated[SupervisorState, InjectedState],
                     tool_call_id: Annotated[str, InjectedToolCallId]
                     ) -> Command:
        
        response_message = f'handoff tool successfully transferred to {next_agent_name}'

        if not state.web_search_content:
            response_message = f"handoff tool Not routing to {next_agent_name} as web_search_content is empty"
        
        tool_message = ToolMessage(
                content=response_message,
                name=next_agent_name,
                tool_call_id=tool_call_id,
        )
        

        return {'messages': [tool_message], 'web_search_content': state.web_search_content}
    #Command(goto='end_subgraph', update={'messages': [tool_message], 'web_search_content': state.web_search_content})

    return handoff_tool