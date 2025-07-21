import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ToolSet, FunctionTool, ListSortOrder

from user_defined_functions import user_functions



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
        if 'function-call-agent' in a.name:
            agent = a
            break

if not agent:
    agent = agent_client.create_agent(
         model=model,
         name="function-call-agent",
         instructions="""You are a helpful assistant, user functions to help answer user questions.
                      """,
         toolset=toolset
     )
    

thread = agent_client.threads.create()

# # convert tempreture from fahrenheit to celsius
# message = agent_client.messages.create(
#     thread_id=thread.id,
#     role="user",
#     content="covert 100 fahrenheit to celsius"
# )

# run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

# messages = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
# for msg in messages:
#     if msg.text_messages:
#         last_text = msg.text_messages[-1]
#         print(f"{msg.role}: {last_text.text.value}")


# get first 2 user info
message = agent_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="get the name of the first 2 users in the user information"
)

run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
for msg in messages:
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")

