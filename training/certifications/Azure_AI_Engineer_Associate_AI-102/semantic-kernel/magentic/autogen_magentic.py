import asyncio
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local  import LocalCommandLineCodeExecutor


async def main() -> None:

    user_prompt = "Provide a different proof for Fermat's Last Theorem"

    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv('PROJECT_ENDPOINT_STRING'),
        model="gpt-4",
        api_version="2024-12-01-preview",
        api_key=os.getenv('openai_key')
    )

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )


    # team = MagenticOneGroupChat([assistant], model_client=model_client)
    # await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))
    # await model_client.close()


    m1 = MagenticOne(client=model_client, code_executor=LocalCommandLineCodeExecutor())
        
    async for chunk in m1.run_stream(task=user_prompt):
        if chunk.__class__.__name__ != 'TaskResult':
            print(chunk.content, end='', flush=True)
        else:
            print(chunk.content, end='', flush=True)


asyncio.run(main())
