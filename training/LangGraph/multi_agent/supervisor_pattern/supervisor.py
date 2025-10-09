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
    
    # this list of agents will be filtered based on completed=True before passing to system prompt
    agents = [
        {"name": "web_search", "description": "search the web for information based on user query", "completed": False},
        {"name": "rag", "description": "perform retrieval augmented generation to get information from vector database", "completed": False},
        {"name": "content_comparer", "description": "compare and merge content from different sources", "completed": False},
        {"name": "END", "description": "only after completing web_search, rag and content_comparer, then lastly route to END", "completed": False}
    ]


    def update_agent_completion(agent_name):
        for agent in agents:
            if agent['name'] == agent_name:
                agent['completed'] = True
        return agents

    
    agents = update_agent_completion("web_search") if state.web_search_content else agents
    agents = update_agent_completion("rag") if state.rag_content else agents
    agents = update_agent_completion("content_comparer") if state.final_content else agents

    agents_to_route = ', \n'.join([f"agent: {agent['name']}, description: {agent['description']}" for agent in agents if not agent['completed']])



    system_prompt = f"""You are a supervisor agent that manages multiple list of agents to accomplish complex tasks. \n
    Your role is to delegate tasks to the appropriate agents. and ensure the overall goal is achieved efficiently.

    The available agents are: {agents_to_route}

    Goal: is to perform web search then use RAG tool to get more information, compare and merge, synthesize or summarize 2 sources into single content.
    Only select content_comparer when both rag and web_search agents have completed their tasks.
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal["rag", "web_search", "content_comparer", "END"]


    # if state.messages and state.messages[0].type != 'system':
    #     state.messages = [SystemMessage(content=system_prompt)] + state.messages

    state.messages = [SystemMessage(content=system_prompt)] + state.messages

    llm = create_llm().with_structured_output(Router)

    response = llm.invoke(state.messages)

    route_to_agent = response['next']

    # routing to different agents without StateGraph deciding next node using add_edge
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
                create_handoff_tool(next_agent_name="end_subgraph", 
                                    description="after all tools complete execution, this last handoff tool to route to next node")
                ])
        
        system_msg = SystemMessage(content="""You are a specialized web search agent tasked with using search tool for web searches.
                        Only use the tool tavily_search_tool to do web search for answers and not from your pre-trained knowledge.
                            After getting internet search result from tool, you must use the handoff tool to delegate task back to supervisor for next steps.
                        """)
        
        if state.messages and state.messages[0].type != 'system':
            state.messages = [system_msg] + state.messages


        last_message = state.messages[-1] if state.messages else None

        messages = [system_msg] + state.messages
        #web_search_content = state.web_search_content # state updated by tavily_search_tool if called

        # check tool name to support multiple tools
        # state.web_search_content_reducer will prevent override of web_search_content with other TooMessage like handoff_tool or other tool message is returned
        # another way to prevent state value override is to have tool update state directly
        # if last_message and isinstance(last_message, ToolMessage) and last_message.name == 'tavily_search_tool':
        #     web_search_content = last_message.content

        #if last_message and not isinstance(last_message, ToolMessage):
        ai_message: AIMessage = llm.invoke(state.messages)
        messages = state.messages + [ai_message]
        
        return {'messages': messages, 'web_search_content': state.web_search_content}
    

    def web_search_end_node(state: SupervisorState) -> Command[str]:
        # do anything before ending this subgraph
        return state


    sub_graph = StateGraph(SupervisorState)
    sub_graph.add_node('web_search_agent', web_search_agent)
    sub_graph.add_node('tools', ToolNode([tavily_search_tool,
                                        create_handoff_tool(next_agent_name="supervisor", 
                                                description="Completed web search, handoff tool to delegate tasks back to supervisor agent.",
                                                graph=Command.PARENT)
                                        ])
                    )
    sub_graph.add_node('end_subgraph', web_search_end_node)
    
    sub_graph.add_conditional_edges('web_search_agent', tools_condition, {'tools': 'tools', '__end__': 'end_subgraph'})
    sub_graph.add_edge('tools', 'web_search_agent')
    sub_graph.add_edge(START, 'web_search_agent')
    sub_graph.set_finish_point('end_subgraph')

    return sub_graph.compile()


def rag_agent_agent(state: SupervisorState) -> Command[str]:

    human_message = state.messages[-1] if state.messages else HumanMessage(content="")

    prompt = f'''
    user query: {human_message.content}\n\n

    only use RAG tool to do semantic search for information from vector database to answer user query.\n
    After getting information from RAG tool, you must end your execution and route back to supervisor for next steps.
    '''
    
    agent = create_react_agent(
        model=create_llm(),
        tools=[
                rag_tool,

                # if routing is required, use handoff tool to delegate task to another agent
                # create_handoff_tool(next_agent_name="agent_to_route_to", 
                #                     description="handoff tool to delegate tasks back to supervisor agent.")
              ],
        prompt=prompt,
        name="rag_agent",
        state_schema=SupervisorState

    )

    response_state = agent.invoke(state)

    return response_state



def content_curator_agent(state: SupervisorState) -> CompiledStateGraph[Any, None, Any, Any]:

    web_search_content = state.web_search_content
    rag_content = state.rag_content

    messages = [
        SystemMessage(content="""You are a content comparer agent that compares and merges, synthesizes or summarizes 2 sources into single content.
                        Only use the 2 sources contents web search content and rag content provided to you to do comparison and merging, curating or summarization."""),
        HumanMessage(content=f"""
                     Given below 2 source content, compare and merge, curate, synthesized or summarize the 2 contents into single content.

                     Web search content: {web_search_content} \n\n 
                     RAG content: {rag_content}. \n\n

                     Output: single curated, summarized, synthesized content.
                     """)
    ]

    llm = create_llm()

    response = llm.invoke(messages)

    return {'messages': messages, 'final_content': response.content}



#TODO
def response_human_in_the_loop(state: SupervisorState) -> Command[Literal["END"]]:
    if not state.final_content:
        return Command(goto='supervisor')

    return {'messages': AIMessage(content=state.final_content)}



graph_builder = StateGraph(SupervisorState)
graph_builder.add_node('supervisor', supervisor) #, destinations=['rag', 'web_search', 'content_comparer', 'response'])
graph_builder.add_node('web_search', create_web_search_agent(), destinations=['supervisor'])
graph_builder.add_node('rag', rag_agent_agent, destinations=['supervisor'])
graph_builder.add_node('content_comparer', content_curator_agent, destinations=['supervisor'])

# supervisor does the routing via Command and child agents routes back to Supervisor via handoff_tool
graph_builder.add_edge(START, 'supervisor')
# graph_builder.add_edge('supervisor', 'web_search')
# graph_builder.add_edge('supervisor', 'rag')
# graph_builder.add_edge('supervisor', 'content_comparer')
graph_builder.add_edge('web_search', 'supervisor')
graph_builder.add_edge('rag', 'supervisor')
graph_builder.add_edge('content_comparer', 'supervisor')
graph_builder.add_edge('supervisor', END)


graph: CompiledStateGraph = graph_builder.compile()

try:

    state = graph.invoke(SupervisorState(
        messages=[HumanMessage(content="In Azure Fundamental concepts, what is the purpose of Application Security Group?")],
        rag_content="",
        web_search_content="",
        final_content=""
    ))

    print(f'Final State: {state["final_content"]}')
    
    # for m in graph.invoke(SupervisorState(
    #     messages=[HumanMessage(content="In Azure Fundamental concepts, what is the purpose of Application Security Group?")],
    #     rag_content="",
    #     web_search_content="",
    #     final_content=""
    # )):
    #     if isinstance(m, dict):
    #         print(f'{m}\n\n')
    #     elif isinstance(m, BaseMessage):
    #         print(f'{m.type}: {m.content}\n\n')
    #     else:
    #         print(f'{m}\n\n')


except Exception as e:
    print(f'Error: {e}')

    
    


    