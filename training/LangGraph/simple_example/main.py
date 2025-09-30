from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal
import random
from langgraph.graph import StateGraph, START, END
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
import os

load_dotenv()

llm = AzureAIChatCompletionsModel(
    model="o4-mini",
    api_version="2024-12-01-preview"
)

result = llm.invoke('Tell me a joke and include some emojis')

print(result)



 
class FlightState(TypedDict):
    name: str
    flight_number: str
    upgrade: Optional[bool] = False
    message: Optional[str]



def check_in(state: FlightState):
    state['message'] = (
        f"checking in {state['name']}",
        f"on flight {state['flight_number']}"
    )
    return state


def check_upgrade(state: FlightState) -> Literal['confirm_check_in', 'offer_upgrade']:

    if random.random() < 0.5:
        state['upgrade'] = True
        return 'offer_upgrade'
    
    state['upgrade'] = False
    return 'confirm_check_in'

def offer_upgrade(state: FlightState) -> bool:
    state['message'] = (
        f"offering upgrade for {state['name']}",
        f"on flight {state['flight_number']}"
    )
    return state

def confirm_check_in(state: FlightState):
    state['message'] = (
        f"No upgrade, confirmed check in for {state['name']}",
        f"on flight {state['flight_number']}"
    )
    return state



builder = StateGraph(FlightState)

builder.add_node('check_in', check_in)
builder.add_node('offer_upgrade', offer_upgrade)
builder.add_node('confirm_check_in', confirm_check_in)

builder.add_edge(START, 'check_in')
builder.add_conditional_edges('check_in', check_upgrade)
builder.add_edge('offer_upgrade', END)
builder.add_edge('confirm_check_in', END)

graph = builder.compile()