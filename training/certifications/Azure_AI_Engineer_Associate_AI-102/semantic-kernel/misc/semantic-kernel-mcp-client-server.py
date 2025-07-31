from semantic_kernel.agents import ChatCompletionAgent, AgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import asyncio

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# https://microsoftlearning.github.io/mslearn-ai-agents/Instructions/03c-use-agent-tools-with-mcp.html
# https://learn.microsoft.com/api/mcp

async def main():

    # connect to MCP server via STDIO

    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "fastmcp_mcp_server.py")

    #Create server parameters
    stdio_server_params = StdioServerParameters(
        command="python",
        args=[server_path]
    )

    # Create the connection via stdio transport
    async with stdio_client(stdio_server_params) as stream:
        async with ClientSession(*stream) as session:

            await session.initialize()

            tools = await session.list_tools()

            print(tools)

    # Initialize the chat completion servi`c`e with Azure OpenAI
    chat_completion_service = AzureChatCompletion(
        deployment_name="gpt-4",
        api_version='2025-01-01-preview',
        api_key=os.getenv("openai_key"),
        endpoint=os.getenv("PROJECT_ENDPOINT_STRING")
    )

    agent = ChatCompletionAgent(
        name='semantic-kernel-agent-hello-world',
        service=chat_completion_service,
        instructions="You are a helpful assistant."
    )

    prompt = "What is the capital of France?"

    response = await agent.get_response(messages=[prompt])

    print(f"Response: {response}")


asyncio.run(main())
