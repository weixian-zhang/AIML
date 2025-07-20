import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose

# project_client = AIProjectClient(
#     subscription_id="c9061bc7-fa28-41d9-a783-2600b29c6e2f",
#     resource_group_name="rg-aiml",
#     project_name="learn-genai",
#     endpoint="https://admin-md73lsw2-eastus2.services.ai.azure.com",
#     credential=DefaultAzureCredential()
#     #conn_str="07385b72-a642-47af-ba98-11d316b79d3f.workspace.eastus.api.azureml.ms;c9061bc7-fa28-41d9-a783-2600b29c6e2f;rg-aiml;learn-genai"
# )

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="07385b72-a642-47af-ba98-11d316b79d3f.workspace.eastus.api.azureml.ms;c9061bc7-fa28-41d9-a783-2600b29c6e2f;rg-aiml;learn-genai"
)

current_file_path = Path(__file__).resolve()
current_directory = os.path.dirname(current_file_path)


# with open(os.path.join(current_directory,"nifty_500_quarterly_results.csv"), "rb")as f:
#     print(len(f.read()))

file = project_client.agents.upload_file_and_poll(
    file=open(os.path.join(current_directory, "nifty_500_quarterly_results.csv"), "rb"),
    purpose='assistants'#FilePurpose.AGENTS
)


code_interpreter = CodeInterpreterTool(file_ids=[file.id])


# create agent with code interpreter tool and tools_resources
# agent = project_client.agents.create_agent(
#     model="gpt-4o-mini",
#     name="my-agent",
#     instructions="You are helpful agent",
#     tools=code_interpreter.definitions,
#     tool_resources=code_interpreter.resources,
# )

# agent already created, so we can retrieve it
agent = project_client.agents.create_agent(
    model="gpt-4o-mini",
    name="finance-agent",
    instructions="You are helpful agent",
    tools=code_interpreter.definitions,
    tool_resources=code_interpreter.resources,
)

thread = project_client.agents.create_thread()

message = project_client.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="create bar chart in the TRANSPORTATION sector for the operating profit from the uploaded csv file and provide file to me?",
)


run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
print(f"Run finished with status: {run.status}")

if run.status == "failed":
    # Check if you got "Rate limit is exceeded.", then you want to get more quota
    print(f"Run failed: {run.last_error}")


# print the messages from the agent
messages = project_client.agents.list_messages(thread_id=thread.id)
print(f"Messages: {messages}")
