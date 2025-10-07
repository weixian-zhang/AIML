# https://blog.futuresmart.ai/multi-agent-system-with-langgraph
import os, sys
# get parent path
langgraph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
supervisor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(langgraph_dir)
sys.path.append(supervisor_dir)

from typing import Literal, TypedDict, Any
# from langchain_core.tools import ToolCall
from langchain_azure_ai import AzureAIChatCompletionsModel
from rag_tool import rag_tool
from web_search_tool import tavily_search_tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from handoff_tool import create_handoff_tool
from state import SupervisorState


load_dotenv()


def create_llm():
    llm = AzureAIChatCompletionsModel(  
        model="o4-mini",
        api_version="2024-12-01-preview",
    )
    return llm
    

def supervisor(state: SupervisorState): #Command[Literal["rag", "web_search","content_comparer", "response"]]:
    # agents = ["rag", "web_search", "content_comparer", "END"]
    agents = ["rag", "END"]
    list_of_agents = ', '.join(agents)

    system_prompt = f"""You are a supervisor agent that manages multiple specialized agents in this list of agents: {list_of_agents}, to accomplish complex tasks. \n
    Your role is to delegate tasks to the appropriate agents. and ensure the overall goal is achieved efficiently.
    Goal: is to perform internet search from human input, use RAG to get more information, compare and merge, synthesize or summarize 2 sources into single content.
    Only select content_comparer when both rag and web_search agents have completed their tasks.
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        # next: Literal["rag", "web_search", "content_comparer", "END"]
        next: Literal["rag", "END"]



    if state.messages and state.messages[0].type != 'system':
        state.messages = [SystemMessage(content=system_prompt)] + state.messages

    llm = create_llm().with_structured_output(Router)

    response = llm.invoke(state.messages)

    route_to_agent = response['next']

    return Command(goto=route_to_agent)



def create_web_search_agent() -> StateGraph[SupervisorState]:
    """
    options to execute tool:
    1. by using LangGraph's "create_react_agent"
    2. create subgraph: agent node + tool node + routing logic
    3. manual tool execution with BaseTool.invoke and return ToolMessage

    this function uses option 2 to create a subgraph, simulating a create_react_agent
    """

    def web_search_agent(state: SupervisorState) -> Command[str]:
        
        llm = create_llm().bind_tools([
                tavily_search_tool, 
                create_handoff_tool(next_agent_name="supervisor", 
                                    description="handoff tool to delegate tasks back to supervisor agent.",
                                    graph=Command.PARENT)
                ])
        
        system_msg = SystemMessage(content="""You are a specialized web search agent tasked with using search tool for web searches.
                        Only use the tool tavily_search_tool to do web search for answers and not from your pre-trained knowledge.
                            After getting internet search result from tool, you must use the handoff tool to delegate task back to supervisor for next steps.
                        """)
        
        if state.messages and state.messages[0].type != 'system':
            state.messages = [system_msg] + state.messages


        last_message = state.messages[-1] if state.messages else None

        messages = state.messages
        web_search_content = state.web_search_content

        # check tool name to support multiple tools
        # state.web_search_content_reducer will prevent override of web_search_content with other TooMessage like handoff_tool or other tool message is returned
        # another way to prevent state value override is to have tool update state directly
        if last_message and isinstance(last_message, ToolMessage) and last_message.name == 'tavily_search_tool':
            web_search_content = last_message.content

        if last_message and not isinstance(last_message, ToolMessage):
            ai_message: AIMessage = llm.invoke(state.messages)
            messages = state.messages + [ai_message]
        
        return {'messages': messages, 'web_search_content': web_search_content}


    builder = StateGraph(SupervisorState)
    builder.add_node('web_search_agent', web_search_agent)
    builder.add_node('tools', ToolNode([tavily_search_tool,
                                        create_handoff_tool(next_agent_name="supervisor", 
                                                description="Completed web search, handoff tool to delegate tasks back to supervisor agent.",
                                                graph=Command.PARENT)]))
    
    builder.add_conditional_edges('web_search_agent', tools_condition, {'tools': 'tools', '__end__': 'web_search_agent'})
    builder.add_edge('tools', 'web_search_agent')
    builder.add_edge(START, 'web_search_agent')

    return builder.compile()



def create_rag_agent() -> CompiledStateGraph[Any, None, Any, Any]:

    system_msg = SystemMessage(content="""You are a helpful assistant tasked with using tools for retrieval-augmented generation.
                                      Only use the tool rag_tool to find answers and not from your pre-trained knowledge.
                                      If you don't know the answer, just say you don't know. Do not make up an answer.
                               """)

    agent = create_react_agent(
        model=create_llm(),
        tools=[
                rag_tool,
                create_handoff_tool(next_agent_name="supervisor", 
                                    description="handoff tool to delegate tasks back to supervisor agent.")
              ], # [rag_tool],
        prompt=system_msg,
        name="rag_agent",
        state_schema=SupervisorState,

    )

    return agent

    # ai_response: AIMessage = agent.invoke(state.messages)

    # if ai_response.tool_calls:
    #     pass

    # content = ai_response.content

    # return Command(update= {
    #     'messages': [
    #         ai_response, #HumanMessage(content=content)
    #     ],
    #     'rag_content': content
    # }, goto="supervisor")


def create_content_comparer_agent() -> CompiledStateGraph[Any, None, Any, Any]:

    # internet_source = state.web_search_content
    # rag_source = state.rag_content

    # if not internet_source or not rag_source:
    #     return Command(goto="supervisor")

    system_msg = SystemMessage(content="""You are a helpful assistant tasked to compare content of 2 sources and merge, \n
                               synthesize, merge or summarize 2 sources into single content.""")
    
    HumanMessage(content="""from Azure Fundamental content 2 sources: 'Internet Source' and 'RAG Source', compare and merge, synthesize or summarize 2 sources into single content" \n
    Internet Source: {internet_source} \n
    RAG Source: {rag_source} \n\n
                 
    Output: a single content that makes the most sense and is accurate based on 2 sources.
    """)

    agent = create_react_agent(
    model=create_llm(),
    tools=[
            rag_tool,
            create_handoff_tool(next_agent_name="supervisor", 
                                description="handoff tool to delegate tasks back to supervisor agent.")
            ], # [rag_tool],
    system_message=system_msg,
    name="create_content_comparer_agent",
    state_schema=SupervisorState,
    )

    return agent


def rag_agent_node(state: SupervisorState) -> Command[str]:

    human_message = state.messages[-1] if state.messages else None

    agent = create_react_agent(
        model=create_llm(),
        tools=[
                rag_tool,
                create_handoff_tool(next_agent_name="supervisor", 
                                    description="handoff tool to delegate tasks back to supervisor agent.")
              ],
        prompt=human_message,
        name="rag_agent",
        state_schema=SupervisorState

    )

    response = agent.invoke(state)

    return {'messages': response['messages']}

#TODO
def response_human_in_the_loop(state: SupervisorState) -> Command[Literal["END"]]:
    if not state.final_content:
        return Command(goto='supervisor')

    return {'messages': AIMessage(content=state.final_content)}



graph_builder = StateGraph(SupervisorState)
graph_builder.add_node('supervisor', supervisor) #, destinations=['rag', 'web_search', 'content_comparer', 'response'])
graph_builder.add_node('web_search', create_web_search_agent(), destinations=['supervisor'])
graph_builder.add_node('rag', rag_agent_node, destinations=['supervisor'])
# graph_builder.add_node('response', response, destinations=[END])
# graph_builder.add_node('content_comparer', create_content_comparer_agent(), destinations=['supervisor'])

graph_builder.add_edge(START, 'supervisor')
# graph_builder.add_edge('supervisor', 'web_search')
graph_builder.add_edge('supervisor', 'rag')
# graph_builder.add_edge('supervisor', 'content_comparer')
graph_builder.add_edge('supervisor', END)


graph = graph_builder.compile()

try:
    for m in graph.stream(SupervisorState(
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
except Exception as e:
    print(f'Error: {e}')

    
    