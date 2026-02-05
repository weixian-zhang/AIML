from typing import Annotated, Literal, List
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STATE: Tracks exploration tree
# =============================================================================

class TreeNode(BaseModel):
    path_id: str  # e.g., "A", "A.1", "A.1.2"
    reasoning: str
    score: float  # How promising this path is (0-1)
    is_solution: bool = False
    is_dead_end: bool = False
    is_visited: bool = False

class ToTState(BaseModel):
    problem: str
    current_path: str = "root"  # Which path we're exploring
    explored_nodes: dict[str, TreeNode] = {}  # All explored paths
    best_solution: TreeNode | None = None
    iteration: int = 0
    max_iterations: int = 10

class DecisionBranch(BaseModel):
    branch_id: str
    reason: str
    confidence: float
    branches: List['DecisionBranch']

class DecisionBranchList(BaseModel):
    branches: List[DecisionBranch]

class BranchEvaluationVerdict(BaseModel):
    is_solution: bool
    is_solution_reason: str
    is_dead_end: bool
    is_dead_end_reason: str
    should_continue: bool
    should_continue_reason: str

# =============================================================================
# LLM
# =============================================================================

llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )

# =============================================================================
# NODE 1: Generate possible next steps (branching)
# =============================================================================

def generate_branches(state: ToTState) -> dict:
    """
    Generate 3 possible reasoning paths from current position
    """
    print(f"\n🌿 BRANCHING from {state.current_path} (iteration {state.iteration})")
    
    current_node = state.explored_nodes.get(state.current_path)
    context = current_node.reasoning if current_node else "Starting point"
    
    prompt = f"""Problem: {state.problem}

Current reasoning path: {context}

Your task:
Generate 3 high-level strategies for structuring the visit order (e.g., shortest routes first, highest-value attractions first, hybrid strategies, etc.).
For each strategy, branch into at least two possible route sequences, estimating total time and value.
Evaluate each branch, noting tradeoffs like time pressure, wasted travel, or low-value stops.

** output **
For each strategy, provide:
1. A brief description
2. Why this might work
3. A confidence score (0-1)
4. for child branches, repeat the above structure. With branch ids reflecting the path (e.g., "1", "1.1", "1.2", "2", "2.1", etc.)

** Important **
Show your reasoning as a tree, where each branching point reflects a distinct line of thought or planning heuristic.
"""
# Format as:
# Option A: [description] | Confidence: [0-X]
# Option B: [description] | Confidence: [0-X]  
# Option C: [description] | Confidence: [0-X]
    
    llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )
    
    response: DecisionBranchList = llm.with_structured_output(DecisionBranchList).invoke([HumanMessage(content=prompt)])
    
    # Parse branches (simplified - in production use structured output)
    #lines = response.content.split('\n')
    branches = {}

    # flatten child branches
    def recurse_branches(branch: DecisionBranch):
        branches[branch.branch_id] = TreeNode(
            path_id=branch.branch_id,
            reasoning=branch.reason,
            score=branch.confidence
        )
        for child in branch.branches:
            print(f"   Generated branch {child.branch_id}: {child.reason[:60]}... (score: {child.confidence})")
            recurse_branches(child)

    for b in response.branches:
        print(f"   Generated branch {b.branch_id}: {b.reason[:60]}... (score: {b.confidence})")
        recurse_branches(b)

    # for b in response.branches:
    #     #branch_id = f"{state.current_path}.{len(branches)+1}" if state.current_path != "root" else str(len(branches)+1)
    #     branches[b.branch_id] = TreeNode(
    #         path_id=b.branch_id,
    #         reasoning=b.reason,
    #         score=b.confidence
    #     )
    
    # for i, line in enumerate(lines[:3], 1):
    #     if 'Option' in line:
    #         branch_id = f"{state.current_path}.{i}" if state.current_path != "root" else str(i)
            
    #         # Extract confidence (simplified parsing)
    #         confidence = 0.5
    #         if 'Confidence:' in line:
    #             try:
    #                 confidence = float(line.split('Confidence:')[1].strip()[:3])
    #             except:
    #                 pass
            
    #         branches[branch_id] = TreeNode(
    #             path_id=branch_id,
    #             reasoning=line,
    #             score=confidence
    #         )
    #         print(f"   Branch {branch_id}: {line[:60]}... (score: {confidence})")
    
    # Add to explored nodes
    explored = dict(state.explored_nodes)
    explored.update(branches)
    
    # ✅ Select best branch to explore next
    if branches:
        best_branch_id = max(branches.keys(), key=lambda k: branches[k].score)
        print(f"   → Moving to explore branch {best_branch_id}")
        next_path = best_branch_id
    else:
        next_path = state.current_path
    
    return {
        "explored_nodes": explored,
        "current_path": next_path,
        "iteration": state.iteration + 1
    }

# =============================================================================
# NODE 2: Evaluate current path (is it a solution? dead end?)
# =============================================================================

def evaluate_path(state: ToTState) -> dict:
    """
    Evaluate if current path is a solution or dead end
    """
    print(f"\n🔍 EVALUATING path {state.current_path}")
    
    
    current_node = state.explored_nodes[state.current_path]
    current_node.is_visited = True

    
    evaluation_prompt = f"""Problem: {state.problem}

Current reasoning: {current_node.reasoning}

Evaluate this reasoning path:
1. It could be a solution but is this the best solution to the problem? (yes/no)
2. Is this a dead end that won't lead to a solution? (yes/no)
3. Should we continue exploring this path? (yes/no)

** Output **:
    is_solution: True or False
    is_solution_reason: brief explanation]
    is_dead_end: True or False
    is_dead_end_reason: [brief explanation]
    should_continue: True or False
    should_continue_reason: [brief explanation]
"""
    
    llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )
    
    response = llm.with_structured_output(BranchEvaluationVerdict).invoke([HumanMessage(content=evaluation_prompt)])
    #verdict = response.content.strip().upper()
    
    

    updated_node = TreeNode(**current_node.dict())

    if response.is_solution:
        updated_node.is_solution = True
        updated_node.reasoning = response.is_solution_reason
        print(f"   ✅ SOLUTION FOUND!")
        return {
            "explored_nodes": {**state.explored_nodes, state.current_path: updated_node},
            "best_solution": updated_node
        }

    elif response.is_dead_end:
        updated_node.is_dead_end = True
        updated_node.reasoning = response.is_dead_end_reason
        print(f"   ❌ DEAD END - need to backtrack")
        return {
            "explored_nodes": {**state.explored_nodes, state.current_path: updated_node}
        }
    
    else:
        updated_node.reasoning = response.should_continue_reason
        print(f"   ⏭️  CONTINUE exploring")
        return {
            "explored_nodes": {**state.explored_nodes, state.current_path: updated_node}
        }

# =============================================================================
# NODE 3: Select next path to explore (Tree Search Strategy)
# =============================================================================

def select_next_path(state: ToTState) -> dict:
    """
    BACKTRACKING LOGIC: Choose most promising unexplored path
    """
    print(f"\n🎯 SELECTING NEXT PATH...")
    
    # Find all unexplored, non-dead-end paths
    candidates = []

    candidates = [val for key, val in state.explored_nodes.items() if not val.is_solution and not val.is_dead_end and not val.is_visited]
    # for path_id, node in state.explored_nodes.items():
    #     if not node.is_solution and not node.is_dead_end:
    #         # Check if this path has been fully explored (has children)
    #         has_children = any(p.startswith(f"{path_id}.") for p in state.explored_nodes.keys())
    #         if not has_children:
    #             candidates.append((path_id, node.score))
    
    if not candidates:
        print("   ⚠️  No more paths to explore!")
        return {"current_path": "END"}
    
    # Select path with highest score (best-first search)
    #next_path = max(candidates, key=lambda x: x[1])[0]
    next_node = max(candidates, key=lambda x: x.score)
    next_path = next_node.path_id
    print(f"   → Selected path {next_path} (score: {state.explored_nodes[next_path].score})")
    
    return {"current_path": next_path}

# =============================================================================
# ROUTER: Decide what to do next
# =============================================================================

def router(state: ToTState) -> Literal["generate", "evaluate", "select", "END"]:
    """
    DECISION: Continue exploring, backtrack, or terminate?
    """
    print(f"\n🔀 ROUTING (iteration {state.iteration})...")
    
    # Check termination conditions
    if state.best_solution:
        print("   ✅ Solution found! → END")
        return "END"
    
    if state.iteration >= state.max_iterations:
        print("   ⏰ Max iterations reached → END")
        return "END"
    
    if state.current_path == "END":
        print("   🏁 No more paths → END")
        return "END"
    
    # If at root or just selected a path, generate branches
    if state.current_path == "root" or state.current_path not in state.explored_nodes:
        print("   → GENERATE branches")
        return "generate"
    
    # all nodes visited
    if all(n.is_visited for n in state.explored_nodes.values()):
        print("   → All nodes visited → GENERATE more branches")
        return "generate"
    
    # Check if current path has been evaluated
    current_node = state.explored_nodes[state.current_path]


    # If not evaluated yet, evaluate it
    if not current_node.is_visited:
        print("   → EVALUATE current path")
        return "evaluate"
    # if not current_node.is_solution and not current_node.is_dead_end:
    #     # Check if we've already generated children for this node
    #     has_children = any(p.startswith(f"{state.current_path}.") for p in state.explored_nodes.keys())
    #     if not has_children:
    #         print("   → EVALUATE current path")
    #         return "evaluate"
    #     else:
    #         print("   → SELECT next path (backtrack)")
    #         return "select"
    
    # Path is dead end or solution, need to select next
    print("   → SELECT next path (backtrack)")
    return "select"

# =============================================================================
# BUILD GRAPH
# =============================================================================

builder = StateGraph(ToTState)

# Add nodes
builder.add_node("generate", generate_branches)
builder.add_node("evaluate", evaluate_path)
builder.add_node("select", select_next_path)

# Flow
builder.add_edge(START, "generate")

builder.add_conditional_edges(
    "generate",
    router,
    {
        "generate": "generate",
        "evaluate": "evaluate",
        "select": "select",
        "END": END
    }
)

builder.add_conditional_edges(
    "evaluate",
    router,
    {
        "generate": "generate",
        "select": "select",
        "END": END
    }
)

builder.add_conditional_edges(
    "select",
    router,
    {
        "generate": "generate",
        "evaluate": "evaluate",
        "END": END
    }
)

graph = builder.compile()

# =============================================================================
# RUN
# =============================================================================

print("="*70)
print("🌳 TREE OF THOUGHTS REASONING")
print("="*70)

problem_1 = """
You are solving a planning puzzle. A traveler wants to visit four cities—A, B, C, and D—in one day.
Each city has:
* A unique attraction with a time cost (1-3 hours).
* A travel time to every other city (varies 30-90 minutes).
* The traveler wants to maximize total attraction value while keeping the total day under 8 hours, including travel.

Provide the optimal route and reasoning steps.
"""

problem_2 = """ 1. Carlos is at the swimming pool. 
				2. He walks to the locker room, carrying a towel. 
				3. He puts his watch in the towel and carries the towel tightly to a lounger at the poolside. 
				4. At the lounger he opens and vigorously shakes the towel, then walks to the snack bar. 
				5. He leaves the towel at the snack bar, then walks to the diving board. 
				6. Later Carlos realises he has has lost his watch. 
				Where is the single most likely location of the watch?
"""

result = graph.invoke(ToTState(
    problem=problem_2
))

print("\n" + "="*70)
print("📊 EXPLORATION RESULTS")
print("="*70)
print(f"Total iterations: {result['iteration']}")
print(f"Paths explored: {len(result['explored_nodes'])}")

if result.get('best_solution'):
    #"Find the optimal route to visit 4 cities: NYC, Boston, Philly, DC, starting from NYC and minimizing total distance"
))

print("\n" + "="*70)
print("📊 EXPLORATION RESULTS")
print("="*70)
print(f"Total iterations: {result['iteration']}")
print(f"Paths explored: {len(result['explored_nodes'])}")

if result.get('best_solution'):
    print(f"\n✅ BEST SOLUTION:")
    print(f"   Path: {result['best_solution'].path_id}")
    print(f"   Reasoning: {result['best_solution'].reasoning}")
else:
    print("\n⚠️  No solution found within iteration limit")

print("\n🗺️  EXPLORATION TREE:")
for path_id, node in sorted(result['explored_nodes'].items()):
    status = "✅ SOLUTION" if node.is_solution else "❌ DEAD END" if node.is_dead_end else "🔍 explored"
    print(f"  {path_id}: {status} (score: {node.score})")