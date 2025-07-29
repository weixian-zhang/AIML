import os
import asyncio
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#tutorial
#https://medium.com/@speaktoharisudhan/build-an-agent-orchestrator-in-python-with-semantic-kernel-bb271d8f32e1
async def main():
    """Main function to run the chat completion service."""
    kernel = Kernel()

    print(os.getenv("openai_key"))

    chat_completion_service = AzureChatCompletion(
        deployment_name="gpt-4",
        api_version='2025-01-01-preview',
        api_key=os.getenv("openai_key"),
        endpoint=os.getenv("PROJECT_ENDPOINT_STRING")
    )


    kernel.add_service(chat_completion_service)


    chat_history = ChatHistory()


    while True:
        user_input = input("enter your message >>>")
        if user_input.lower() == "q":
            print("You pressed 'q' exiting the program")
            break

        chat_history.add_user_message(user_input)

        response  = await chat_completion_service.get_chat_message_content(
            user_input=user_input,
            kernel=kernel,
            chat_history=chat_history,
            settings= OpenAIChatPromptExecutionSettings(top_p=0.1, temperature=0.7, max_tokens=1000)
        )

        response = str(response)

        print(f"Response: {response}")

asyncio.run(main())