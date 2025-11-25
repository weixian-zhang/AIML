from langgraph.graph import MessagesState, StateGraph, add_messages, START, END
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Image
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, List
from dotenv import load_dotenv
load_dotenv()


llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )


class GraphMessagesState(BaseModel):
    reflection_iteration: int # type: ignore
    generated_advise: str # type: ignore
    reflected_advise: str # type: ignore
    messages: Annotated[list[BaseMessage], add_messages]


def generate(state: GraphMessagesState) -> GraphMessagesState:

    if not type(state.messages[-1]) is HumanMessage:
        return state
    
    system_prompt = SystemMessage(content="You are an expert makeup artist and aesthetic coach. Provide short thoughtful and concise advise not more then 3 sentences.")

    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessage(content=state.messages[-1].content)
    ])

    chain = prompt | llm

    response: AIMessage = chain.invoke({})

    return {'generated_advise': response.content, 'messages': response}


def reflection(state: GraphMessagesState) -> GraphMessagesState:
    if not state.generated_advise:
        return state
    
    system_prompt = SystemMessage(content="You are an expert makeup artist and aesthetic mentor. Reflect on makeup artist students advise and give a thoughtful and short and concise advance not more than 3 sentences.")
    
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessage(content=f"Here is some advise: {state.generated_advise}. Please reflect on it and provide your thoughts.")
    ])

    chain = prompt | llm
    
    response: AIMessage = chain.invoke({})


    return {'reflected_advise': response.content, 
            'reflection_iteration': state.reflection_iteration + 1,
            'messages': [HumanMessage(content=response.content)] }


def should_continue(state: GraphMessagesState) -> bool:
    if state.reflection_iteration >= 2:
        return END
    return "generate"


sg = StateGraph(GraphMessagesState)
sg.add_node('generate', generate)
sg.add_node('reflection', reflection)
sg.add_edge(START, 'generate')
sg.add_edge('generate', 'reflection')
sg.add_conditional_edges('reflection', should_continue)

graph = sg.compile()

# print(graph.get_graph().draw_ascii())

response: AIMessage = graph.invoke({
    'reflection_iteration': 0,
    'generated_advise': '',
    'reflected_advise': '',
    'messages': [HumanMessage(content="how do I make myself look cool?")]
})

print(response['generated_advise'])
