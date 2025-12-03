from typing import Annotated, Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv()

# just need a short explanation to my quesrtion about your example, not need to generate further codes.

# can I say the "iterative reasoning" is done by langgraph as long as the "reason" node which uses LLM to determine if all necessary info is present, then returns 

# =============================================================================
# STATE: Tracks what information we have
# =============================================================================

class PlanningState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Information gathered
    destination: str = ""
    dates: str = ""
    budget: str = ""
    interests: str = ""
    travelers: str = ""
    
    # Agent's reasoning
    missing_info: list[str] = []
    is_ready_to_plan: bool = False
    final_plan: str = ""

# =============================================================================
# AGENT: Analyzes what's missing and decides next question
# =============================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def reasoning_agent(state: PlanningState) -> dict:
    """
    CORE REASONING: Agent thinks about what's missing and what to ask next
    """
    
    print("\n🧠 AGENT THINKING...")
    
    # Analyze current state
    has_destination = bool(state.destination)
    has_dates = bool(state.dates)
    has_budget = bool(state.budget)
    has_interests = bool(state.interests)
    has_travelers = bool(state.travelers)
    
    # Identify missing information
    missing = []
    if not has_destination:
        missing.append("destination")
    if not has_dates:
        missing.append("dates/duration")
    if not has_budget:
        missing.append("budget")
    if not has_interests:
        missing.append("interests/activities")
    if not has_travelers:
        missing.append("number of travelers")
    
    print(f"   ✓ Has: destination={has_destination}, dates={has_dates}, budget={has_budget}")
    print(f"   ✗ Missing: {missing}")
    
    # Decision logic
    if not missing:
        print("   → Decision: ALL INFO COLLECTED, ready to create plan")
        return {
            "is_ready_to_plan": True,
            "missing_info": []
        }
    
    # Agent decides what to ask next (prioritize destination > dates > budget)
    next_question = missing[0]
    print(f"   → Decision: Need to ask about {next_question}")
    
    # Generate contextual question
    system_prompt = f"""You are a helpful travel planning assistant.
    
CURRENT INFORMATION COLLECTED:
- Destination: {state.destination or 'NOT PROVIDED'}
- Dates: {state.dates or 'NOT PROVIDED'}
- Budget: {state.budget or 'NOT PROVIDED'}
- Interests: {state.interests or 'NOT PROVIDED'}
- Travelers: {state.travelers or 'NOT PROVIDED'}

MISSING: {next_question}

Ask the user ONLY about the missing information: {next_question}
Be friendly and conversational. Ask ONE question at a time.
"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state.messages[-3:]  # Last 3 messages for context
    ])
    
    print(f"   💬 Agent asks: {response.content[:60]}...")
    
    return {
        "messages": [response],
        "missing_info": missing,
        "is_ready_to_plan": False
    }

# =============================================================================
# EXTRACTOR: Parse user's response to extract information
# =============================================================================

def extract_info(state: PlanningState) -> dict:
    """
    EXTRACTION: Pull out structured information from user's message
    """
    
    print("\n🔍 EXTRACTING INFORMATION...")
    
    last_user_message = state.messages[-1].content
    
    extraction_prompt = f"""Extract travel planning information from this message:
"{last_user_message}"

CURRENT STATE:
- Destination: {state.destination or 'unknown'}
- Dates: {state.dates or 'unknown'}
- Budget: {state.budget or 'unknown'}
- Interests: {state.interests or 'unknown'}
- Travelers: {state.travelers or 'unknown'}

Extract any NEW information mentioned. Return JSON:
{{
    "destination": "city/country name or empty string",
    "dates": "travel dates or empty string",
    "budget": "budget amount or empty string",
    "interests": "activities/interests or empty string",
    "travelers": "number of people or empty string"
}}

Only fill in fields if EXPLICITLY mentioned in the message.
"""
    
    class ExtractionOutput(BaseModel):
        destination: str = ""
        dates: str = ""
        budget: str = ""
        interests: str = ""
        travelers: str = ""
    
    structured_llm = llm.with_structured_output(ExtractionOutput)
    extracted = structured_llm.invoke([HumanMessage(content=extraction_prompt)])
    
    # Merge with existing state (don't overwrite non-empty fields)
    updates = {}
    if extracted.destination and not state.destination:
        updates["destination"] = extracted.destination
        print(f"   ✓ Extracted destination: {extracted.destination}")
    if extracted.dates and not state.dates:
        updates["dates"] = extracted.dates
        print(f"   ✓ Extracted dates: {extracted.dates}")
    if extracted.budget and not state.budget:
        updates["budget"] = extracted.budget
        print(f"   ✓ Extracted budget: {extracted.budget}")
    if extracted.interests and not state.interests:
        updates["interests"] = extracted.interests
        print(f"   ✓ Extracted interests: {extracted.interests}")
    if extracted.travelers and not state.travelers:
        updates["travelers"] = extracted.travelers
        print(f"   ✓ Extracted travelers: {extracted.travelers}")
    
    if not updates:
        print("   ⚠ No new information extracted")
    
    return updates

# =============================================================================
# PLANNER: Create final holiday plan
# =============================================================================

def create_plan(state: PlanningState) -> dict:
    """
    PLANNING: Generate comprehensive holiday plan
    """
    
    print("\n📋 CREATING FINAL PLAN...")
    
    planning_prompt = f"""Create a detailed holiday plan based on:

DESTINATION: {state.destination}
DATES: {state.dates}
BUDGET: {state.budget}
INTERESTS: {state.interests}
TRAVELERS: {state.travelers}

Generate a comprehensive plan with:
1. Suggested itinerary (day-by-day)
2. Accommodation recommendations
3. Must-see attractions matching interests
4. Budget breakdown
5. Travel tips

Be specific and actionable.
"""
    
    response = llm.invoke([SystemMessage(content=planning_prompt)])
    
    print("   ✓ Plan created!")
    
    return {
        "final_plan": response.content,
        "messages": [AIMessage(content=f"Here's your holiday plan:\n\n{response.content}")]
    }

# =============================================================================
# ROUTER: Decide next step based on state
# =============================================================================

def router(state: PlanningState) -> Literal["extract", "plan", "reason"]:
    """
    DECISION: Where should we go next?
    """
    
    print("\n🔀 ROUTING...")
    
    # If we have all info, create the plan
    if state.is_ready_to_plan:
        print("   → Route to: CREATE PLAN")
        return "plan"
    
    # If last message is from user, extract info first
    if state.messages and isinstance(state.messages[-1], HumanMessage):
        print("   → Route to: EXTRACT INFO")
        return "extract"
    
    # Otherwise, agent needs to ask more questions
    print("   → Route to: ASK MORE QUESTIONS")
    return "reason"

# =============================================================================
# BUILD GRAPH
# =============================================================================

builder = StateGraph(PlanningState)

# Add nodes
builder.add_node("extract", extract_info)
builder.add_node("reason", reasoning_agent)
builder.add_node("plan", create_plan)

# Define flow
builder.add_edge(START, "reason")  # Start with agent reasoning

builder.add_conditional_edges(
    "reason",
    router,
    {
        "extract": "extract",
        "plan": "plan",
        "reason": "reason"
    }
)

builder.add_edge("extract", "reason")  # After extraction, agent reasons again
builder.add_edge("plan", END)  # Planning is final step

graph = builder.compile()

# =============================================================================
# RUN INTERACTIVE SESSION
# =============================================================================

print("="*70)
print("🌴 HOLIDAY PLANNING ASSISTANT")
print("="*70)

state = PlanningState(
    messages=[HumanMessage(content="I want to plan a holiday")]
)

# Simulate conversation
user_inputs = [
    "I want to plan a holiday",
    "I'm thinking Paris",
    "Maybe 5 days in December",
    "Around $2000",
    "I love art and food",
    "Just me and my partner, so 2 people"
]

for user_input in user_inputs:
    print(f"\n{'='*70}")
    print(f"👤 USER: {user_input}")
    print(f"{'='*70}")
    
    # Add user message
    state.messages.append(HumanMessage(content=user_input))
    
    # Run graph until it asks next question or finishes
    result = graph.invoke(state)
    state = PlanningState(**result)
    
    # Show agent's response
    if state.messages and isinstance(state.messages[-1], AIMessage):
        agent_response = state.messages[-1].content
        print(f"\n🤖 AGENT: {agent_response[:200]}...")
    
    # Check if done
    if state.is_ready_to_plan and state.final_plan:
        print("\n" + "="*70)
        print("✅ PLANNING COMPLETE!")
        print("="*70)
        print(state.final_plan)
        break