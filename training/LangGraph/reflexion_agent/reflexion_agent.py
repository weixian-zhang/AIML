

# %%
import os
import json
import getpass
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph, MessageGraph, MessagesState
from dotenv import load_dotenv
load_dotenv()

# %%
tavily_tool=TavilySearchResults(max_results=1)
# sample_query = "healthy breakfast recipes"
# search_results = tavily_tool.invoke(sample_query)
# print(search_results)



# llm = ChatOpenAI(model="gpt-4.1-nano")
llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )

##### responder

system_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalates, lectins, and phytates.

        Your response must follow these steps:
        1. {first_instruction}
        2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
        3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
        4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
        5. List 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.

        Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system", 
        "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
    ),
])


system_first_responder_prompt = system_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
# temp_chain = system_first_responder_prompt| llm
# response = temp_chain.invoke({"messages": [HumanMessage(content=question)]})
# print(response.content)

class GraphState(MessagesState):
    question: str
    answer: str
    reflection: str
    search_queries: List[str] 
    search_query_results: Dict[str, str] = Field(description="Results from search queries")
    references: List[str] = Field(description="Citations motivating your answer.")
    iterate: int = 3

class Reflection(BaseModel):
	missing: str = Field(description="What information is missing")
	superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
	answer: str = Field(description="Main response to the question")
	reflection: Reflection = Field(description="Self-critique of the answer")
	search_queries: List[str] = Field(description="Queries for additional research")
    


initial_chain = system_first_responder_prompt | llm.with_structured_output(AnswerQuestion) #llm.bind_tools(tools=[AnswerQuestion])


##### execute tool

tavily_tool=TavilySearchResults(max_results=3)

def execute_tools(state: GraphState) -> List[BaseMessage]:
    query_results = {}

    search_queries = state['search_queries']

    for query in search_queries:
        result = tavily_tool.invoke(query)
        query_results[query] = result

    return { "search_query_results": query_results }

    # for tool_call in last_ai_message.tool_calls:
    #     if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
    #         call_id = tool_call["id"]
    #         search_queries = tool_call["args"].get("search_queries", [])
    #         query_results = {}
    #         for query in search_queries:
    #             result = tavily_tool.invoke(query)
    #             query_results[query] = result
    #         tool_messages.append(ToolMessage(
    #             content=json.dumps(query_results),
    #             tool_call_id=call_id)
    #         )
    #return tool_messages


##### revisor

revise_instructions = """Revise your previous answer using the new information, applying the rigor and evidence-based approach of Dr. David Attia.
- Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
- You MUST include numerical citations referencing research result: <research_result>, randomized controlled trials, or meta-analyses to ensure medical accuracy.
- Distinguish between correlation and causation, and acknowledge limitations in current research.
- Address potential biomarker considerations (lipid panels, inflammatory markers, and so on) when relevant.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
- [1] https://example.com
- [2] https://example.com
- Use the previous critique to remove speculation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
- When discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
"""
system_revisor_prompt = system_prompt_template.partial(first_instruction=revise_instructions)



class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(description="Citations motivating your updated answer.")

# revisor chain
revisor_chain = system_revisor_prompt | llm.with_structured_output(ReviseAnswer) #llm.bind_tools(tools=[ReviseAnswer])


##### nodes
def responder(state: GraphState) -> dict:
    human_question = state['question']

    """Initial responder node that provides a draft answer."""
    response = initial_chain.invoke({"messages": [HumanMessage(content=human_question)]})

    return {
        "answer": response.answer,
        "reflection": response.reflection,
        "search_queries": response.search_queries,
        'messages': AIMessage(content=response.answer)
    }


def revisor(state: GraphState) -> dict:
    """Revisor node that refines the initial answer."""

    prompt = ChatPromptTemplate.from_messages(
         
         [
        system_revisor_prompt,

        HumanMessage(content="""
        You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition.
        
        1. Based on user question <question>, improve on <previous answer to improve improve> with <previous critique> and Internet research result in <research_result>.
        2. update previous critique to reflect on what was added or removed in the new answer.
        3. List 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.
    
        <question>
        {question}
        </question>   
                                                                                                
        <previous answer to improve>
        {answer}
        </previous answer to improve>
                                                        
        <previous critique missing superfluous>
        {reflection}
        <previous critique missing superfluous>
                     
        <previous search queries>
        {search_queries}
        </previous search queries>

        <research_result>
        {search_query_results}
        </research_result>
        """)
    ])

    revisor_chain = prompt | llm.with_structured_output(ReviseAnswer)


    response = revisor_chain.invoke({"question": state['question'],
                                    "answer": state['answer'],
                                    "reflection": state['reflection'].missing + "; \n\n" + state['reflection'].superfluous,
                                    "search_queries": state['search_queries'],
                                    "search_query_results": state['search_query_results'],
                                    "messages": state['messages']
                                    })
    
    link_references = response.references

    return {
        "answer": response.answer,
        "reflection": response.reflection,
        "messages": AIMessage(content=response.answer),
        "search_queries": response.search_queries,
        "references": link_references,
        'iterate': int(state['iterate']) - 1
    }


MAX_ITERATIONS = 3

def event_loop(state: List[BaseMessage]) -> str:
    # count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    # num_iterations = count_tool_visits
    # if num_iterations >= MAX_ITERATIONS:
    #     return END
    # return "execute_tools"
    if state['iterate'] <= 0:
        return END
    return "execute_tools"


##### define graph

graph= StateGraph(GraphState) #MessageGraph()

graph.add_node("respond", responder)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor)

# %%
graph.add_edge("respond", "execute_tools")
graph.add_edge("execute_tools", "revisor")

# %%
graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("respond")


app = graph.compile()
responses = app.invoke({
     'question': """I'm pre-diabetic and need to lower my blood sugar, and I have heart issues.
    What breakfast foods should I eat and avoid""", 
    'iterate': MAX_ITERATIONS}
)


##### print output

# print("--- Initial Draft Answer ---")
# initial_answer = responses[1].tool_calls[0]['args']['answer']
# print(initial_answer)
# print("\n")

# print("--- Intermediate and Final Revised Answers ---")
# answers = []

# # Loop through all messages in reverse to find all tool_calls with answers
# for msg in reversed(responses):
#     if getattr(msg, 'tool_calls', None):
#         for tool_call in msg.tool_calls:
#             answer = tool_call.get('args', {}).get('answer')
#             if answer:
#                 answers.append(answer)

# Print all collected answers
for i, ai_msg in enumerate(responses['messages']):
    print(f'{i}: {ai_msg.content}\n\n')




