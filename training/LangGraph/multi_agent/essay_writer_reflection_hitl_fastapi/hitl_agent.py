
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_azure_ai  import AzureAIChatCompletionsModel
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    essay_to_review: str = Field(default_factory=str)


llm = AzureAIChatCompletionsModel(  
        model="o4-mini",
        api_version="2024-12-01-preview",
    )


# messages  = [
#     SystemMessage(content="You are a helpful assistant"),
#     HumanMessage(content="What is a good business name'"),
#     AIMessage(content="A good business name could be 'Agentic AyEye'"),
# ]


def essay_generator(state: AgentState):

    return {'messages': state.messages, 'essay_to_review': 'an essay about Transformer'}


def ask_human_feedback(state: AgentState):

    
    value = interrupt({
        "essay_to_review": state.essay_to_review
    })

    print(f"Received human feedback: {value}")

    # When resumed, this will contain the human's input
    return {
        "essay_to_review": value
    }


def receive_human_feedback(state: AgentState):

    print(f'user has gave feedback: {state.essay_to_review}')

    return {'messages': state.messages, 'essay_to_review': state.essay_to_review }


memory_checkpointer = InMemorySaver()

builder = StateGraph(AgentState)
builder.add_node('essay_writer', essay_generator)
builder.add_node('ask_human_feedback', ask_human_feedback)
builder.add_node('receive_human_feedback', receive_human_feedback)
builder.add_edge(START, 'essay_writer')
builder.add_edge('essay_writer', 'ask_human_feedback')
builder.add_edge('ask_human_feedback', 'receive_human_feedback')
builder.add_edge('receive_human_feedback', END)


graph = builder.compile(checkpointer=memory_checkpointer)


config = {"configurable": {"thread_id": '1'}}

for event in graph.stream(AgentState(
    messages= [HumanMessage('''
I like a written topic about transformer architecture
''')]
), config=config):
    
    print(event)


current_state = graph.get_state(config)
interrupt_value = graph.get_state(config).tasks[0].interrupts[0]
print(f"Current graph state: {current_state.values}")
print(f"Next node to execute: {current_state.next}")


final_result = graph.invoke(Command(resume="the essay does not tell the real inner workings of how transformer work under the hood"), config=config)




# https://shaveen12.medium.com/langgraph-human-in-the-loop-hitl-deployment-with-fastapi-be4a9efcd8c0