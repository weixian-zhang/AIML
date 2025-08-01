from semantic_kernel.agents import ChatCompletionAgent, AgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel
import os
import asyncio

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class EmailPlugin:
    """
    A plugin to handle email-related tasks.
    """

    @kernel_function(description='Sends an email to the specified recipient with the given subject and body.')
    async def send_email(self, to: str, subject: str, body: str) -> str:
        """
        Sends an email to the specified recipient with the given subject and body.
        """
        # Here you would implement the logic to send an email.
        # For demonstration purposes, we will just return a success message.
        print(f"Email sent to {to} with subject '{subject}' and body '{body}'")
    

async def main():
    kernel = Kernel()

    kernel.add_plugin(
    EmailPlugin(),
    plugin_name="Email",
    )

    kernel.add_service(AzureChatCompletion(
        deployment_name="gpt-4",  
        api_key=os.getenv("openai_key"),
        endpoint=os.getenv('PROJECT_ENDPOINT_STRING'), # Used to point to your service
        service_id="azure_chat_completion",
    ))


    agent = ChatCompletionAgent(name="EmailAgent", 
                            instructions="You are an email assistant. You can send emails on behalf of the user.",
                            kernel=kernel)


    response = await agent.get_response(
        messages=["Send an email to Weixian with Subject 'Hello World Semantic Kernel' and content or body 'This is a test email.'"])

    print(f"Response: {response}")



asyncio.run(main())
