from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langgraph.graph.message import add_messages
from datetime import datetime
from IPython.display import Image, display
from web_search_tool import tavily_search_tool

load_dotenv()

 # https://ai.google.dev/gemini-api/docs/langgraph-example
 
class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def route_tools(state: AgentState):
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"  # Route to the tool execution node
    else:
        return END  # End the graph if no tool calls


def reasoner(state: AgentState):
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
                                      
    state.messages = [system_msg] + state.messages
    response = llm.invoke(state.messages)
    return {'messages': [response] }

# returns a dict[str, BaseTool]
tools = {tool.name: tool for tool in [tavily_search_tool] }

def tool_executor(state: AgentState):
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_call = last_message.tool_calls[-1]  # Get the latest tool call
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']

        tool = tools[tool_name]


        observation = tool.invoke(tool_args)
        observation_msg = ToolMessage(observation, tool_call_id=tool_id) #AIMessage(content=f"Observation: {observation}")
        state.messages.append(observation_msg)

    return {'messages': state.messages}


def should_call_tools(state: AgentState) -> bool:
    last_message = state.messages[-1]
    if not last_message.tool_calls:
        return END
    return "tools"


# LLM
llm = AzureAIChatCompletionsModel(
    model="o4-mini",
    api_version="2024-12-01-preview",
).bind_tools([tavily_search_tool])


tool_node = ToolNode([tavily_search_tool])

graph_builder = StateGraph(AgentState)
graph_builder.add_node('reasoner', reasoner)
graph_builder.add_node('tools', tool_executor)

graph_builder.add_edge(START, 'reasoner')
graph_builder.add_conditional_edges('reasoner',
                                    
                                    # if latest message from result is tool call -> tool_condition routes tools
                                    # if latest message from result is not a tool call, tools_condition routes to END
                                    should_call_tools) #tools_condition)
graph_builder.add_edge('tools', 'reasoner')

react_graph = graph_builder.compile()

result = react_graph.invoke({
    "messages": [HumanMessage(content="what is the latest news in Singapore today?")]
})

for m in result['messages']:
    print(f'{m.type}: {m.content}')