from typing_extensions import Annotated
from typing import Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import SystemMessage

# class AgentState(BaseModel):
#     messages: Annotated[list[BaseMessage], add_messages]



def web_search_content_reducer(current_value: str, new_value: str) -> str:
    # Implement your custom logic here
    if not current_value:
        return new_value
    return current_value


class SupervisorState(BaseModel):
    remaining_steps: int = Field(default=2)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    rag_content: Optional[str] = Field(default='')
    web_search_content: Annotated[str, web_search_content_reducer] = Field(default_factory='')
    final_content: Optional[str] = Field(default='')
    id: int = Field(default=0)

    """
    set default message value in StateGraph throws unhasable error, so we set default in __hash__ method
    """
    def __hash__(self):
        return hash(self.id)
    

# class WebSearchSubGraphState(BaseModel):
#     messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
#     web_search_content: Optional[str] = Field(default='')
#     id: int = Field(default=0)

#     """
#     set default message value in StateGraph throws unhasable error, so we set default in __hash__ method
#     """
#     def __hash__(self):
#         return hash(self.id)