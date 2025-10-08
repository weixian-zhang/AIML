from langchain_tavily import TavilySearch
from langchain.tools import BaseTool, tool
from langgraph.types import Command
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from state import SupervisorState
from langgraph.graph import add_messages


@tool(description='Web search tool that searches the web for information given a query.', return_direct=False)
def tavily_search_tool(query: str, include_domains: list[str], 
                       state: Annotated[SupervisorState, InjectedState],
                       tool_call_id: Annotated[str, InjectedToolCallId]) -> Annotated[SupervisorState, add_messages]:
    '''web searcher'''

    include_domains = [
        'https://www.channelnewsasia.com/',
        'https://www.straitstimes.com/'
    ]
    
    search_client = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=True,
        # include_images=False,
        # include_image_descriptions=False, 
        search_depth="advanced",
        # time_range="day",
        include_domains=include_domains
        # exclude_domains=None
    )

    tool_message = ToolMessage(
        content=f"Successfully used web search tool to get information from the web.",
        name="tavily_search_tool",
        tool_call_id=tool_call_id
    )

    answer =  "ASG is an abstraction to list of VM private IP addresses making up a logical group for easier management."

    return Command(goto="web_search", update={"messages": [tool_message], "web_search_content": answer})
# {"messages": [tool_message], "web_search_content": answer}
#

    #


        # search_result = search_client.invoke({"query": query, "include_domains": include_domains})

        # answer =  "ASG is an abstraction to list of VM private IP addresses making up a logical group for easier management." # search_result['answer']

        # # set state like this does not work, injected state is immutable
        # state.web_search_content = answer

        # return answer
    




