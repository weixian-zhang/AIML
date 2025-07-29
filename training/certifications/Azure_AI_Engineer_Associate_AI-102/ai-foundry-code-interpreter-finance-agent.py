import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
# from azure.ai.projects import AIProjectClient
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, AgentEventHandler, ListSortOrder


class StreamEventHandler(AgentEventHandler[str]):

    def on_message_delta(self, delta) -> str:
        return f"Text delta received: {delta.text}"

    def on_thread_message(self, message) -> str:
        return f"ThreadMessage created. ID: {message.id}, Status: {message.status}"

    def on_thread_run(self, run) -> str:
        return f"ThreadRun status: {run.status}"

    def on_run_step(self, step) -> str:
        return f"RunStep type: {step.type}, Status: {step.status}"

    def on_error(self, data: str) -> str:
        return f"An error occurred. Data: {data}"

    def on_done(self) -> str:
        return "Stream completed."

    def on_unhandled_event(self, event_type: str, event_data) -> str:
        return f"Unhandled Event Type: {event_type}, Data: {event_data}"



project_endpoint = os.environ.get("PROJECT_ENDPOINT_STRING")
current_file_path = Path(__file__).resolve()
current_directory = os.path.dirname(current_file_path)

agent = None

agent_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential()
)


agents = agent_client.list_agents()
if agents:
    for i, a in enumerate(agents):
        if 'finance-agent' in a.name:
            agent = a


file = agent_client.files.upload_and_poll(
    file=open(os.path.join(current_directory, "nifty_500_quarterly_results.csv"), "rb"),
    purpose=FilePurpose.AGENTS
)


code_interpreter = CodeInterpreterTool(file_ids=[file.id])



if not agent:
    agent = agent_client.create_agent(
        model="gpt-4o",
        name="finance-agent",
        instructions="You are an AI agent that analyzes the data in the file that has been uploaded. Use Python to calculate statistical metrics as necessary.",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )

thread = agent_client.threads.create()

# Create a message
message = agent_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="Could you please create bar chart in TRANSPORTATION sector for the operating profit from the uploaded csv file and provide file to me?"
)


run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
for msg in messages:
    if msg.image_contents:
        for image_content in msg.image_contents:
            print(f"Image File ID: {image_content.image_file.file_id}")
            file_name = f"{image_content.image_file.file_id}_image_file.png"
            agent_client.files.save(file_id=image_content.image_file.file_id, file_name=file_name, target_dir=current_directory)
            print(f"Saved image file to: {Path.cwd() / file_name}")
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")

