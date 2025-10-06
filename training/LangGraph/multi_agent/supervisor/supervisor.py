# https://blog.futuresmart.ai/multi-agent-system-with-langgraph
import os, sys
# get parent path
langgraph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
supervisor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(langgraph_dir)
sys.path.append(supervisor_dir)

from typing import Literal, TypedDict
# from langchain_core.tools import ToolCall
from langchain_azure_ai import AzureAIChatCompletionsModel
# from tools.rag_tool import rag_tool
from tools.web_search_tool import tavily_search_tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from handoff_tool import create_handoff_tool
from state import AgentState



load_dotenv()


class SupervisorState(AgentState):
    rag_content: str
    web_search_content: str
    final_content: str


def create_llm():
    llm = AzureAIChatCompletionsModel(  
        model="o4-mini",
        api_version="2024-12-01-preview",
    )
    return llm
    

def supervisor(state: AgentState): #Command[Literal["rag", "web_search","content_comparer", "response"]]:
    agents = ["rag", "web_search", "content_comparer", "response"]
    list_of_agents = ', '.join(agents)

    system_prompt = f"""You are a supervisor agent that manages multiple specialized agents in this list of agents: {list_of_agents}, to accomplish complex tasks. \n
    Your role is to delegate tasks to the appropriate agents. and ensure the overall goal is achieved efficiently.
    Goal: is to perform internet search from human input, use RAG to get more information, compare and merge, synthesize or summarize 2 sources into single content.
    Only select content_comparer when both rag and web_search agents have completed their tasks.
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal["rag", "web_search", "content_comparer", "END"]


    messages = [SystemMessage(content=system_prompt)] + state.messages

    llm = create_llm().with_structured_output(Router)

    response = llm.invoke(messages)

    route_to_agent = response['next']

    return Command(goto=route_to_agent)



# Use the following format:\n
# Thought: Do I need to use a tool? Yes\n
# Action: the action to take, should be one of [tavily_search_tool]\n
# Action Input: the input to the action\n
# Observation: the result of the action\n
# ... (this Thought/Action/Action Input/Observation can repeat N times)\n
# Thought: Do I need to use a tool? No\n
# Final Answer: the final answer to the original question.\n
# Only use the tools get_weather, tavily_search_tool, add. 
# If you don't know the answer, just say you don't know. Do not make up an answer.

def create_web_search_agent() -> StateGraph[AgentState]:
    """
    options to execute tool:
    1. by using LangGraph's "create_react_agent"
    2. create subgraph: agent node + tool node + routing logic
    3. manual tool execution with BaseTool.invoke and return ToolMessage
    """

    def web_search_agent(state: AgentState) -> Command[str]:

        system_msg = SystemMessage(content="""You are a helpful assistant tasked with using search tool for internet searches.
                                Only use the tool tavily_search_tool to do web search for answers and not from your pre-trained knowledge.
                                   After getting internet search result from tool, you must use the handoff tool to delegate task back to supervisor for next steps.
                                """)
        
        llm = create_llm().bind_tools([
                tavily_search_tool, 
                create_handoff_tool(next_agent_name="supervisor", 
                                    description="handoff tool to delegate tasks back to supervisor agent.",
                                    graph=Command.PARENT)
                ])

        messages = [system_msg] + state.messages

        ai_message: AIMessage = llm.invoke(messages)

        return {'messages': ai_message}
    

    # def tool_router(state: AgentState) -> Command[str]:
    #     last_message = state.messages[-1] if state.messages else None

    #     if isinstance(last_message, AIMessage) and last_message.tool_calls:
    #         tool_name = last_message['name']
    #         tool_args = last_message['args']
    #         tool_id = last_message['id']
    #         if tool_name == 'tavily_search_tool':
    #             return AIMessage()

    builder = StateGraph(AgentState)
    builder.add_node('web_search_agent', web_search_agent)
    builder.add_node('tools', ToolNode([tavily_search_tool]))
    builder.add_node('handoff', ToolNode([create_handoff_tool(next_agent_name="supervisor", 
                                                description="Completed web search, handoff tool to delegate tasks back to supervisor agent.",
                                                graph=Command.PARENT)]))
    builder.add_conditional_edges('web_search_agent', tools_condition, {'tools': 'tools', 'END': 'web_search_agent'})
    builder.add_edge('tools', 'web_search_agent')
    builder.add_edge(START, 'web_search_agent')
    builder.add_edge('web_search_agent', 'handoff')

    return builder.compile()



def rag_agent(state: AgentState) -> Command[Literal["supervisor"]]:

    system_msg = SystemMessage(content="""You are a helpful assistant tasked with using tools for retrieval-augmented generation.
                                      Only use the tool rag_tool to find answers and not from your pre-trained knowledge.
                                      If you don't know the answer, just say you don't know. Do not make up an answer.
                               """)

    llm = create_llm().bind_tools([rag_tool])


    messages = [system_msg] + state.messages

    ai_response: AIMessage = llm.invoke(messages)

    if ai_response.tool_calls:
        pass

    content = ai_response.content

    return Command(update= {
        'messages': [
            ai_response, #HumanMessage(content=content)
        ],
        'rag_content': content
    }, goto="supervisor")


def content_comparer_agent(state: AgentState) -> Command[Literal["supervisor"]]:

    internet_source = state.web_search_content
    rag_source = state.rag_content

    if not internet_source or not rag_source:
        return Command(goto="supervisor")

    system_msg = SystemMessage(content="""You are a helpful assistant tasked with comparing content of 2 sources and merge, synthesize or summarize 2 sources into single content.""")
    
    HumanMessage(content="""from Azure Fundamental content 2 sources: 'Internet Source' and 'RAG Source', compare and merge, synthesize or summarize 2 sources into single content" \n
    Internet Source: {internet_source} \n
    RAG Source: {rag_source} \n
    """)

    llm = create_llm()

    messages = [system_msg] + state.messages

    ai_response: AIMessage | ToolMessage = llm.invoke(messages)


    return Command(update= {
        'messages': [
            ai_response ##HumanMessage(content=content)
        ],
        'final_content': ai_response.content
    }, goto="supervisor")


def response(state: AgentState) -> Command[Literal["END"]]:
    if not state.final_content:
        return Command(goto='supervisor')

    return {'messages': AIMessage(content=state.final_content)}



graph_builder = StateGraph(AgentState)
graph_builder.add_node('supervisor', supervisor) #, destinations=['rag', 'web_search', 'content_comparer', 'response'])
graph_builder.add_node('web_search', create_web_search_agent())
# graph_builder.add_node('rag', rag_agent, destinations=['supervisor'])
graph_builder.add_node('response', response, destinations=[END])
graph_builder.add_node('content_comparer', content_comparer_agent, destinations=['supervisor'])

graph_builder.add_edge(START, 'supervisor')
graph_builder.add_edge('supervisor', 'web_search')
# graph_builder.add_edge('supervisor', 'rag')
# graph_builder.add_edge('supervisor', 'content_comparer')
graph_builder.add_edge('supervisor', END)


graph = graph_builder.compile()

for m in graph.stream(AgentState(
    messages=[HumanMessage(content="In Azure Fundamental concepts, what is the purpose of Application Security Group?")],
    rag_content="",
    web_search_content="",
    final_content=""
)):
    if isinstance(m, dict):
        print(f'{m}\n\n')
    elif isinstance(m, BaseMessage):
        print(f'{m.type}: {m.content}\n\n')
    else:
        print(f'{m}\n\n')

    
    