from semantic_kernel.agents import ChatCompletionAgent, AgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments, KernelPlugin
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import asyncio

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# https://microsoftlearning.github.io/mslearn-ai-agents/Instructions/03c-use-agent-tools-with-mcp.html
# https://medium.com/@nvsmanoj0202/semantic-kernels-contextual-challenges-overcoming-limitations-with-the-model-context-protocol-d0be54bd1e0c

async def main():

    # connect to MCP server via STDIO

    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "fastmcp_mcp_server.py")

    try:
        ## using mcp module
        # stdio_server_params = StdioServerParameters(
        #     command="python",
        #     args=[server_path]
        # )

        # # Create the connection via stdio transport
        # async with stdio_client(stdio_server_params) as stream:
        #     async with ClientSession(*stream) as session:

        #         await session.initialize()

        #         tools = await session.list_tools()

        #         plugin = KernelPlugin()
        #         plugin.add_list(tools)
                
        #         prompts = await session.list_prompts()
                
        #         agent = ChatCompletionAgent(
        #             name='semantic-kernel-agent-hello-world',
        #             service= AzureChatCompletion(
        #                 deployment_name="gpt-4",
        #                 api_version='2025-01-01-preview',
        #                 api_key=os.getenv("openai_key"),
        #                 endpoint=os.getenv("PROJECT_ENDPOINT_STRING")
        #             ),
        #             instructions="You are a helpful assistant.",
        #             plugins=[plugin]
        #         )


    
        ava_mcp_plugins = MCPStdioPlugin(
                    name="AVA",
                    description="external service invoker",
                    command="python",
                    args=[server_path]
                )
    
        await ava_mcp_plugins.connect()
        

        agent = ChatCompletionAgent(
                name='semantic-kernel-agent-hello-world',
                service= AzureChatCompletion(
                    deployment_name="gpt-4",
                    api_version='2025-01-01-preview',
                    api_key=os.getenv("openai_key"),
                    endpoint=os.getenv("PROJECT_ENDPOINT_STRING")
                ),
                instructions="You are a helpful assistant.",
                plugins=[ava_mcp_plugins]
            )

    
        prompt = "Do invoke the external service to find out more."

        response = await agent.get_response(messages=[prompt])

        print(f"Response: {response.message.content}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        await ava_mcp_plugins.close()


asyncio.run(main())
