import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ToolSet, FunctionTool, ListSortOrder

from user_defined_functions import user_functions

'''
Submits support ticket using function call
'''


functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)

agent = None
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
        if 'support-ticket-agent' in a.name:
            agent = a
            break

if not agent:
    agent = agent_client.create_agent(
        model=model,
        name="support-ticket-agent",
        instructions="""You are a technical support agent.
                         When a user has a technical issue, you get their email address and a description of the issue.
                         Then you use those values to submit a support ticket using the function available to you.
                         If a file is saved, tell the user the file name.
                     """,
        toolset=toolset,
    )
    

thread = agent_client.threads.create()


message = agent_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="I have a technical issue with my Azure subscription." \
    "My email is user@example.com and the issue is not being able to access the portal. " \
    "Please submit a support ticket for me."
)

run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
for msg in messages:
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")