from langchain_tavily import TavilySearch
from langchain.tools import BaseTool, tool

@tool
def tavily_search_tool(query: str, include_domains: list[str]) -> BaseTool:
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

    

    tool_msg = tool.invoke({"query": query, "include_domains": include_domains})

    return tool_msg['answer']

