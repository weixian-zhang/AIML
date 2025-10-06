from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from langgraph.graph import add_messages

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]