import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool, ToolSet, FunctionTool, ListSortOrder
from dotenv import load_dotenv, find_dotenv
from user_defined_functions import user_functions

# Find the .env file by searching parent directories
dotenv_path = find_dotenv()
# Load environment variables from the found .env file
load_dotenv(dotenv_path)

functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)

function_call_agent = None
support_ticket_agent = None
model = 'gpt-4o'
project_endpoint = os.environ.get("PROJECT_ENDPOINT_STRING")

agent_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential()
)

agent_client.enable_auto_function_calls(toolset)

agents = agent_client.list_agents()
if agents:
    for i, a in enumerate(agents):
        if 'function_call_agent' in a.name:
            function_call_agent = a
            continue
        if 'support_ticket_agent' in a.name:
            support_ticket_agent = a
        

if not function_call_agent:
    function_call_agent = agent_client.create_agent(
         model=model,
         name="function_call_agent",
         instructions="""You are a helpful assistant, use functions to help answer user questions.""",
         toolset=toolset
     )
    

if not support_ticket_agent:
    support_ticket_agent = agent_client.create_agent(
        model=model,
        name="support_ticket_agent",
        instructions="""You are a technical support agent.
                         When a user has a technical issue, you get their email address and a description of the issue.
                         Then you use those values to submit a support ticket using the function available to you.
                         If a file is saved, tell the user the file name.""",
        toolset=toolset,
    )

connected_func_call_agent = ConnectedAgentTool(id=function_call_agent.id, 
                                               name=function_call_agent.name, 
                                               description=function_call_agent.instructions)

connected_support_ticket_agent = ConnectedAgentTool(id=support_ticket_agent.id,
                                                   name=support_ticket_agent.name,
                                                   description=support_ticket_agent.instructions)


# orchestrator_toolset = ToolSet()
# orchestrator_toolset.add(connected_func_call_agent)
# orchestrator_toolset.add(connected_support_ticket_agent)

orchestrator_agent = agent_client.create_agent(
    model=model,
    name="orchestrator_agent",
    instructions="""You are an orchestrator agent.
                     You can call the function call agent to answer user questions.
                     You can also call the support ticket agent to submit a support ticket.
                     """,
    tools=[connected_func_call_agent.definitions[0], 
           connected_support_ticket_agent.definitions[0]]
)


thread = agent_client.threads.create()

message = agent_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="""I have a technical issue with my computer. My email is user@example.com and the issue is that it won't turn on.
    Please submit a support ticket for me."""
)

run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=orchestrator_agent.id)


messages = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
for msg in messages:
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")


agent_client.threads.delete(thread.id)
