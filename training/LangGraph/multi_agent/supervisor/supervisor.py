# https://blog.futuresmart.ai/multi-agent-system-with-langgraph
import os, sys
# get parent path
langgraph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(langgraph_dir)

from typing import Literal, Annotated, TypedDict
from langchain_azure_ai import AzureAIChatCompletionsModel
from tools.rag_tool import rag_tool
from tools.web_search_tool import tavily_search_tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    result: str
    

def create_llm():
    llm = AzureAIChatCompletionsModel(  
        model="o4-mini",
        api_version="2024-12-01-preview",
    )

    return llm
    

def supervisor(state: AgentState) -> Command[Literal["rag", "web_search", "END"]]:
    agents = ["rag", "web_search"]
    list_of_agents = ', '.join(agents)

    system_prompt = f"""You are a supervisor agent that manages multiple specialized agents in this list of agents: {list_of_agents}, to accomplish complex tasks. \n
    Your role is to delegate tasks to the appropriate agents, monitor their progress, and ensure the overall goal is achieved efficiently.
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal["rag", "web_search", "END"]


    messages = SystemMessage(content=system_prompt) + state.messages

    llm = create_llm().with_structured_output(Router)

    response = llm.invoke(messages)

    route_to_agent = response['next']

    return Command(goto=route_to_agent)




def web_search_agent(state: AgentState) -> Command[Literal["supervisor"]]:
    system_msg = SystemMessage(content="""You are a helpful assistant tasked with using tools for latest news.
                                      Use the following format:\n
                                      Thought: Do I need to use a tool? Yes\n
                                      Action: the action to take, should be one of [tavily_search_tool]\n
                                      Action Input: the input to the action\n
                                      Observation: the result of the action\n
                                      ... (this Thought/Action/Action Input/Observation can repeat N times)\n
                                      Thought: Do I need to use a tool? No\n
                                      Final Answer: the final answer to the original question.\n
                                      Only use the tools get_weather, tavily_search_tool, add. 
                                      If you don't know the answer, just say you don't know. Do not make up an answer.
                               """)

    llm = create_llm().bind_tools([tavily_search_tool])

    messages = [system_msg] + state.messages

    response = llm.invoke({'messages': messages})

    return Command(update= {
        'messages': [
            HumanMessage(content=response['messages'][-1].content)
        ]
    }, goto="supervisor")
                                      
    


    
    