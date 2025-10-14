
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_azure_ai  import AzureAIChatCompletionsModel
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
import pprint
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    topic: str = Field(default_factory=str)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    essay_output: str = Field(default_factory=str)
    plan: str = Field(default_factory=str)
    web_research_content: str = Field(default_factory=str)
    human_feedback: str = Field(default_factory=str)


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

def writer_system_prompt(web_research_content: str) -> str:

    return '''
    You are an essay assistant tasked with writing excellent 10-paragraph essays.\n
    Generate the best essay possible for the user's request and the initial outline. \n
    If the user provides critique, respond with a revised version of your previous attempts. \n
    Utilize all web research content information below as needed: 
    \n\n
    ------
    web research content:\n
    {web_research_content}
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

def web_researcher_system_prompt(plan: str) -> str:
    return f'''
    You are a researcher charged with providing information that can \n
    be used when writing the following essay topic or plan. Generate a list of search queries that will gather \n
    any relevant information.\n\n

    Only generate 3 queries max.\n\n

    \n\n
    -----
    essay_topic:\n
    {plan}
    '''


def reflect_on_human_feedback():
    return f'''
    You are an expert essay writer reviewing draft essay and user's feedback.\n
    Take user feedback into serious consideration and use user's feedback to make amendments to the draft.
    '''


human_feedback_sentiment_system_prompt = """determine if user's is positive or negative. If positive means user approves of essay and negative sentiment means user rejects draft essay.\n
final output: Outputs only 'approve' or 'reject'."""


def planner(state: AgentState):

    messages = [SystemMessage(content= planner_system_prompt()),
                HumanMessage(content=state.topic)]

    response: AIMessage = llm.invoke(messages)

    return { 'messages': messages + [response], 'plan': response.content}


def web_researcher(state: AgentState):

    class Query(TypedDict):
        query: list[str]

    if not state.plan:
        return Command(goto='planner')
    
    research_content = []
    messages = [SystemMessage(content=web_researcher_system_prompt(state.plan)),
                HumanMessage(content=state.topic)]

    queries = llm.with_structured_output(Query).invoke(messages)


    for q in queries['query']:
        result = search_client.invoke({'query': q})
        content =  result['answer']
        research_content.append(content)

    final_research_content = '\n\n'.join(research_content)
    
    return {'messages': messages, 'web_research_content': final_research_content}


def writer(state: AgentState):


    messages = [
        SystemMessage(content=writer_system_prompt(state.web_research_content)),
        HumanMessage(content=f'''this is the essay topic:{state.topic} \n\n
                     my plan: {state.plan}''')
    ]

    response: AIMessage = llm.invoke(messages)

    return {'messages': messages + [response], 'essay_output': response.content}



def human_feedback(state: AgentState):

    human_feedback = interrupt({
        "essay_output": state.essay_output
    })


    # When resumed, this will contain the human's input
    return {
        "human_feedback": human_feedback
    }

def human_approve_reject(state: AgentState):

    # class ApproveReject(BaseModel):
    #     approve_reject: Literal["approve", "reject"]

    # response: AIMessage = llm.with_structured_output(ApproveReject).invoke([
    #     SystemMessage(content=human_feedback_sentiment_system_prompt),
    #     HumanMessage(content=state.human_feedback)
    # ])

    # if (approve_reject := response['approve_reject']) == 'approve':
    #     return END

    if 'approve' in state.human_feedback.lower():
        return END
    
    return 'reflection_on_human_feedback'



def reflection_on_human_feedback(state: AgentState):
    human_feedback = state.human_feedback
    draft_essay = state.essay_output

    response: AIMessage = llm.invoke([
        SystemMessage(content=reflect_on_human_feedback()),
        HumanMessage(content=f'''Revise draft essay based on user's feedback\n\n
                     
                     user feedback:\n
                     {human_feedback}\n\n

                    ------\n

                    draft essay:\n
                    {draft_essay}
                     ''')
    ])


    return { 'messages': state.messages + [response], 'essay_output': response.content }




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
builder.add_edge('reflection_on_human_feedback', 'human_feedback')


graph = builder.compile(checkpointer=memory_checkpointer)


config = {"configurable": {"thread_id": '1'}}


for event in graph.stream(AgentState(
    topic = '''I like a written topic about transformer architecture'''
), config=config, stream_mode=['values']):
    
    print([f'{k}: {v[10:]}...{v[-10:]}\n' for k, v in event[1].items()])

loop = 0
num_of_feedback = 5

while loop < num_of_feedback:

    current_state = graph.get_state(config)

    if not current_state.tasks:
        print('Graph completed')
        break

    
    essay = ''
    if current_state.tasks[0].interrupts:
        interrupt_data = current_state.tasks[0].interrupts[0].value['essay_output']
        print(f"Interrupt data: {interrupt_data}")
        essay = graph.get_state(config=config).tasks[0].interrupts[0].value

    pprint.pprint(essay)

    user_feedback = input('your feedback')

    for event in graph.invoke(
            Command(resume=user_feedback),
            config=config
        ):
        # node_name = list(event.keys())[0]
        # print(f"   Executed: {node_name}")
        #pprint.pprint(event)
        pass

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