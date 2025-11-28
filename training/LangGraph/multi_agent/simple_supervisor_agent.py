from typing import Annotated, Literal
from pydantic import BaseModel
from langchain_azure_ai import AzureAIChatCompletionsModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# State
class AgentState(BaseModel):
    task: str
    researcher_output: str = ""
    analyst_output: str = ""
    writer_output: str = ""
    final_report: str = ""
    next_agent: str = ""

# LLM
llm = AzureAIChatCompletionsModel(
    model="o4-mini",
    api_version="2024-12-01-preview",
)

# Supervisor: Decides which agent to route to
def supervisor(state: AgentState) -> dict:
    """Routes to next agent or synthesis"""
    
    # If all agents have completed, go to synthesizer
    if state.researcher_output and state.analyst_output and state.writer_output:
        return {"next_agent": "synthesizer"}
    
    # Route to agents that haven't completed yet
    if not state.researcher_output:
        return {"next_agent": "researcher"}
    elif not state.analyst_output:
        return {"next_agent": "analyst"}
    elif not state.writer_output:
        return {"next_agent": "writer"}
    
    return {"next_agent": "synthesizer"}

# Agent 1: Researcher
def researcher(state: AgentState) -> dict:
    """Conducts research on the task"""
    print("🔍 Researcher working...")
    
    response: AIMessage = llm.invoke([
        SystemMessage(content="You are a research specialist. Gather key facts and information."),
        HumanMessage(content=f"Research this topic: {state.task}")
    ])
    
    return {"researcher_output": response.content}

# Agent 2: Analyst
def analyst(state: AgentState) -> dict:
    """Analyzes the research findings"""
    print("📊 Analyst working...")
    
    response: AIMessage = llm.invoke([
        SystemMessage(content="You are a data analyst. Analyze information and find insights."),
        HumanMessage(content=f"Analyze this research: {state.researcher_output}\n\nOriginal task: {state.task}")
    ])
    
    return {"analyst_output": response.content}

# Agent 3: Writer
def writer(state: AgentState) -> dict:
    """Creates written content"""
    print("✍️ Writer working...")
    
    response: AIMessage = llm.invoke([
        SystemMessage(content="You are a professional writer. Create clear, engaging content."),
        HumanMessage(content=f"Write about: {state.task}\n\nBased on analysis: {state.analyst_output}")
    ])
    
    return {"writer_output": response.content}

# Synthesizer: Consolidates all agent outputs
def synthesizer(state: AgentState) -> dict:
    """Combines all agent outputs into final report"""
    print("🔄 Synthesizer consolidating results...")
    
    response: AIMessage = llm.invoke([
        SystemMessage(content="You are a synthesis specialist. Combine multiple perspectives into a cohesive report."),
        HumanMessage(content=f"""Create a comprehensive final report by synthesizing these three perspectives:

        RESEARCH FINDINGS:
        {state.researcher_output}

        ANALYSIS:
        {state.analyst_output}

        WRITTEN CONTENT:
        {state.writer_output}

        Original task: {state.task}
        """)
    ])
    
    return {"final_report": response.content}

# Router function for conditional edges
def route_supervisor(state: AgentState) -> Literal["researcher", "analyst", "writer", "synthesizer"]:
    """Routes based on supervisor's decision"""
    return state.next_agent

# Build graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("analyst", analyst)
builder.add_node("writer", writer)
builder.add_node("synthesizer", synthesizer)

# Add edges
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "synthesizer": "synthesizer"
    }
)

# After each agent completes, return to supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("writer", "supervisor")

# Synthesizer is the final node
builder.add_edge("synthesizer", END)

# Compile
graph = builder.compile()

# Run
result = graph.invoke(AgentState(
    task="Explain the benefits of AI agents in enterprise applications"
))

print("\n" + "="*50)
print("📋 FINAL REPORT:")
print("="*50)
print(result['final_report'])