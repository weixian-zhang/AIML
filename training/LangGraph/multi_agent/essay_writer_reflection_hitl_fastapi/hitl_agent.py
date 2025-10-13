
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_azure_ai  import AzureAIChatCompletionsModel
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    essay_output: str = Field(default_factory=str)
    plan: str = Field(default_factory=str)
    web_research_content: str = Field(default_factory=str)
    user_feedback: str = Field(default_factory=str)


llm = AzureAIChatCompletionsModel(  
        model="o4-mini",
        api_version="2024-12-01-preview",
    )

search_client = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=True,
        # include_images=False,
        # include_image_descriptions=False, 
        search_depth="advanced",
        # time_range="day",
        # include_domains=include_domains
        # exclude_domains=None
    )


def planner_system_prompt() -> str:
    return '''
    You are an expert writer tasked with writing a high level outline of an essay. \n
    Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \n
    or instructions for the sections.
    '''

def writer_prompt(research_content: str) -> str:

    return '''
    You are an essay assistant tasked with writing excellent 10-paragraph essays.\n
    Generate the best essay possible for the user's request and the initial outline. \n
    If the user provides critique, respond with a revised version of your previous attempts. \n
    Utilize all research_content information below as needed: 
    \n\n
    ------
    research_content:\n
    {research_content}
    '''

def reflection_prompt(essay: str) -> str: 
    return f'''
    You are a teacher grading an essay submission. \n
    Generate critique, suggestions and recommendations for student's essay below. \n
    Provide detailed recommendations, including requests for length, depth, style, etc.\n
    \n\n
    -----
    student essay:\n
    {essay}
    '''

def reflect_on_human_feedback(human_feedback: str):
    return f'''
    You are an expert essay writer reviewing draft essay and user's feedback.\n
    Take user feedback into serious consideration and use. user feedback to make amendments to the draft.\n
    \n\n
    -----
    user feedback:
    {human_feedback}
    '''

def web_researcher_system_prompt(plan: str) -> str:
    return f'''
    You are a researcher charged with providing information that can \n
    be used when writing the following essay topic or plan. Generate a list of search queries that will gather \n
    any relevant information. Only generate 3 queries max.
    \n\n
    -----
    essay_topic:\n
    {plan}
    '''

human_feedback_sentiment_system_prompt = """determine if user's is positive or negative. If positive means user approves of essay and negative sentiment means user rejects draft essay.\n
final output: Outputs only 'approve' or 'reject'."""


def planner(state: AgentState):
    system_prompt = SystemMessage(content= planner_system_prompt())

    messages = [system_prompt] + state.messages

    response: AIMessage = llm.invoke(messages)

    return { 'messages': [response], 'plan': response.content}


def web_researcher(state: AgentState):

    class Query(TypedDict):
        query: str

    if not state.plan:
        return Command(goto='planner')
    
    research_content = []
    system_prompt = SystemMessage(content=web_researcher_system_prompt(state.plan))

    queries = llm.with_structured_output(Query).invoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=state.plan)    # plan from planner node
    ])

    for q in queries:
        result = search_client.invoke({'query': q})
        content =  result['answer']
        research_content.append(content)

    final_research_content = '\n'.join(research_content)
    
    return {'web_research_content': final_research_content}


def writer(state: AgentState):

    system_prompt = SystemMessage(content=writer_prompt(state.web_research_content))
    human_prompt = f'my plan: {state.plan}'

    response: AIMessage = llm.invoke([system_prompt, human_prompt])

    return {'messages': state.messages, 'essay_output': response.content}



def human_feedback(state: AgentState):

    human_feedback = interrupt({
        "essay_output": state.essay_output
    })


    # When resumed, this will contain the human's input
    return {
        "human_feedback": human_feedback
    }

def human_approve_reject(state: AgentState):

    class ApproveReject:
        approve_reject = Literal['approve', 'reject']

    response: AIMessage = llm.with_structured_output(ApproveReject).invoke([
        SystemMessage(content=human_feedback_sentiment_system_prompt),
        HumanMessage(content=state.human_feedback)
    ])

    if (approve_reject := response.content) == 'approve':
        return END
    
    return 'human_feedback'

     


def reflection_on_human_feedback(state: AgentState):
    human_feedback = state.human_feedback
    system_prompt = reflect_on_human_feedback(human_feedback=human_feedback)

    response: AIMessage = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.essay_output)
    ])

    return { 'messages': state.messages, 'essay_output': response.content }


def reflection(state: AgentState) -> dict:
    user_feedback = state.user_feedback





memory_checkpointer = InMemorySaver()

builder = StateGraph(AgentState)
builder.add_node('planner', planner)
builder.add_node('web_researcher', web_researcher)
builder.add_node('writer', writer)
builder.add_node('human_feedback', human_feedback)
builder.add_node('reflection_on_human_feedback', reflection_on_human_feedback)
builder.add_edge(START, 'planner')
builder.add_edge('planner', 'web_researcher')
builder.add_edge('web_researcher', 'writer')
builder.add_edge('writer', 'human_feedback')
builder.add_conditional_edges('human_feedback', human_approve_reject)


graph = builder.compile(checkpointer=memory_checkpointer)


config = {"configurable": {"thread_id": '1'}}


for event in graph.stream(AgentState(
    messages= [HumanMessage(content='''I like a written topic about transformer architecture''')]
), config=config):
    
    print(event)


loop = 0
num_of_feedback = 5

while loop < num_of_feedback:


    current_state = graph.get_state(config)

    if not current_state.tasks:
        print('Graph completed')
        break

    
    essay = ''
    if current_state.tasks[0].interrupts:
        interrupt_data = current_state.tasks[0].interrupts[0].value
        print(f"Interrupt data: {interrupt_data}")
        essay = graph.get_state(config=config).tasks[0].interrupts[0].value

    print(essay)

    user_feedback = input('your feedback')

    for event in graph.stream(
            Command(resume_value=user_feedback),
            config=config
        ):
        node_name = list(event.keys())[0]
        print(f"   Executed: {node_name}")
        print(event)

    # Check if approved
    final_state = graph.get_state(config)
    if final_state.values.get('approved'):
        print("\n🎉 Essay approved! Exiting feedback loop.")
        break


    loop += 1


# interrupt_value = graph.get_state(config).tasks[0].interrupts[0]
# print(f"Current graph state: {current_state.values}")
# print(f"Next node to execute: {current_state.next}")


#final_result = graph.invoke(Command(resume="the essay does not tell the real inner workings of how transformer work under the hood"), config=config)




# https://shaveen12.medium.com/langgraph-human-in-the-loop-hitl-deployment-with-fastapi-be4a9efcd8c0