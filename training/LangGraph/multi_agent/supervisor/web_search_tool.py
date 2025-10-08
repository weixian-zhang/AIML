from langchain_tavily import TavilySearch
from langchain.tools import BaseTool, tool
from langgraph.types import Command
from typing import Annotated
from langgraph.prebuilt import InjectedState
from state import SupervisorState
from langgraph.graph import add_messages


def create_web_search_tool(next_agent: str):
    
    @tool(description='Web search tool that searches the web for information given a query.', return_direct=False)
    def tavily_search_tool(query: str, include_domains: list[str], state: Annotated[SupervisorState, InjectedState]) -> Annotated[SupervisorState, add_messages]:
        '''web searcher'''

        include_domains = [
            'https://www.channelnewsasia.com/',
            'https://www.straitstimes.com/'
        ]
        
        tool = TavilySearch(
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

        return Command(goto=next_agent, 
                       update={
                        'messages': state.messages,
                        'web_search_content': state.web_search_content
                       }
        )


        # # tool_msg = tool.invoke({"query": query, "include_domains": include_domains})

        # #state.web_search_content = tool_msg['answer']
        
        # answer =  "ASG is an abstraction to list of VM private IP addresses making up a logical group for easier management."

        # # set state like this does not work, injected state is immutable
        # state.web_search_content = answer

        # return answer
    

    return tavily_search_tool


