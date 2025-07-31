from fastmcp import FastMCP
import os

mcp = FastMCP('AVA')

@mcp.prompt(name='get_prompt_1')
def get_prompt_1():
    return '''
    You are a helpful assistant but a sarcastic one.

    **Preferences:**
    - You should always respond with a sarcastic tone.
    - You should always respond in a clear and concise way.
    '''

@mcp.resource('resource://90s-movies-recommendation')
def get_90s_movies_recommendation():
    fp = os.path.dirname(__file__)
    with open(fp) as f:
        return f.read()
    

@mcp.tool(name='invoke_external_service', description='Invoke an external service')
def invoke_external_service(prompt: str):
    # Simulate an external service call
    return f"External service response for prompt: {prompt}"



if __name__ == "__main__":
    mcp.run()


