import asyncio
from fastmcp import FastMCP
import os

mcp = FastMCP('AVA')

@mcp.prompt(name='get_prompt_1')
def get_prompt_1() -> str:
    return '''
    You are a helpful assistant but a sarcastic one.

    **Preferences:**
    - You should always respond with a sarcastic tone.
    - You should always respond in a clear and concise way.
    '''

@mcp.resource('resource://90s-movies-recommendation')
def get_90s_movies_recommendation() -> str:
    fp = os.path.join(os.path.dirname(__file__), '90s_films.csv')
    with open(fp) as f:
        return f.read()
    

@mcp.tool(name='invoke_external_service', description='Invoke an external service')
async def invoke_external_service() -> str: #(prompt: str) -> str:
    # Simulate an external service call
    #return f"External service response for prompt: {prompt}"
    return f"External service executed successfully."


async def main():
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())


