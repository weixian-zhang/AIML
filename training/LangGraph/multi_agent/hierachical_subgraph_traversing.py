

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict

# Define state
class State(TypedDict):
    messages: list[str]
    action: str

# ============================================
# LEVEL 3: DEEPEST CHILD SUBGRAPH
# ============================================
def level3_process(state: State):
    """Processes at the deepest level"""
    msg = "    [Level 3] Deep processing complete"
    print(msg)
    return {"messages": state["messages"] + [msg]}

def create_level3_subgraph():
    """Creates the deepest subgraph (Level 3)"""
    graph = StateGraph(State)
    graph.add_node("process", level3_process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()

# ============================================
# LEVEL 2: MIDDLE SUBGRAPH
# ============================================
def level2_analyze(state: State):
    """Analyzes and decides routing"""
    msg = "  [Level 2] Analyzing request..."
    print(msg)
    return {"messages": state["messages"] + [msg]}

def level2_route(state: State):
    """Routes from Level 2 to Level 3 child OR up to Level 1 parent"""
    action = state.get("action", "").lower()
    
    if "deep" in action:
        # Route to Level 3 CHILD subgraph (sibling node in same graph)
        # NO Command.PARENT - we're going to a child, not parent!
        print("  [Level 2] Routing to Level 3 child subgraph...")
        return Command(
            update={"messages": state["messages"] + ["  [Level 2] Going deeper..."]},
            goto="level3_child"  # This is a child subgraph node at current level
            # NO graph=Command.PARENT here!
        )
    elif "escalate" in action:
        # Route UP to Level 1 PARENT graph
        print("  [Level 2] Escalating to Level 1 parent...")
        return Command(
            update={"messages": state["messages"] + ["  [Level 2] Escalating up..."]},
            goto="escalation",  # Node in Level 1 parent graph
            graph=Command.PARENT  # YES - we're going UP to parent
        )
    else:
        # Stay at Level 2
        msg = "  [Level 2] Processing at current level"
        print(msg)
        return {"messages": state["messages"] + [msg]}

def create_level2_subgraph():
    """Creates the middle subgraph (Level 2)"""
    graph = StateGraph(State)
    graph.add_node("analyze", level2_analyze)
    graph.add_node("route", level2_route)
    graph.add_node("level3_child", create_level3_subgraph())  # Level 3 as child
    
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "route")
    graph.add_edge("route", END)  # Default flow
    graph.add_edge("level3_child", END)  # After Level 3 completes
    
    return graph.compile()

# ============================================
# LEVEL 1: PARENT GRAPH (TOP LEVEL)
# ============================================
def level1_start(state: State):
    """Entry point"""
    msg = "[Level 1] Starting workflow"
    print(msg)
    return {"messages": [msg]}

def level1_escalation(state: State):
    """Handles escalations from Level 2"""
    msg = "[Level 1] ESCALATION HANDLER - Processing urgent case"
    print(msg)
    return {"messages": state["messages"] + [msg]}

def level1_finalize(state: State):
    """Final step"""
    msg = "[Level 1] Workflow complete"
    print(msg)
    return {"messages": state["messages"] + [msg]}

def create_level1_graph():
    """Creates the top-level parent graph (Level 1)"""
    graph = StateGraph(State)
    graph.add_node("start", level1_start)
    graph.add_node("level2_sub", create_level2_subgraph())  # Level 2 as child
    graph.add_node("escalation", level1_escalation)
    graph.add_node("finalize", level1_finalize)
    
    graph.add_edge(START, "start")
    graph.add_edge("start", "level2_sub")
    graph.add_edge("level2_sub", "finalize")
    graph.add_edge("escalation", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()

# ============================================
# EXAMPLES
# ============================================
app = create_level1_graph()

print("=" * 70)
print("Example 1: Level 2 → Level 3 (DOWN to child) - NO Command.PARENT")
print("=" * 70)
result1 = app.invoke({"action": "deep process", "messages": []})
print("\nExecution path:")
for msg in result1["messages"]:
    print(f"{msg}")

print("\n" + "=" * 70)
print("Example 2: Level 2 → Level 1 (UP to parent) - WITH Command.PARENT")
print("=" * 70)
result2 = app.invoke({"action": "escalate urgent", "messages": []})
print("\nExecution path:")
for msg in result2["messages"]:
    print(f"{msg}")

print("\n" + "=" * 70)
print("Example 3: Stay at Level 2 - NO Command at all")
print("=" * 70)
result3 = app.invoke({"action": "normal process", "messages": []})
print("\nExecution path:")
for msg in result3["messages"]:
    print(f"{msg}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("NAVIGATION SUMMARY:")
print("=" * 70)
print("""
Graph Structure:
  Level 1 (Parent)
    └── Level 2 (Subgraph)
          └── Level 3 (Child Subgraph)

From Level 2, you can route:

1. DOWN to Level 3 (child):
   Command(goto="level3_node")  
   ❌ NO Command.PARENT
   
2. UP to Level 1 (parent):
   Command(goto="level1_node", graph=Command.PARENT)
   ✅ YES Command.PARENT
   
3. Stay at Level 2 (sibling):
   Command(goto="level2_node")
   ❌ NO Command.PARENT

Rule: Command.PARENT only when going UP the hierarchy!
""")