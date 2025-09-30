from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal
import random
import langgraph

load_dotenv()
 
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


def check_upgrade(state: FlightState) -> Literal['confirmed_check_in', 'offer_upgrade']:

    if random.random() < 0.5:
        state['upgrade'] = True
        return 'offer_upgrade'
    
    return 'confirmed_check_in'