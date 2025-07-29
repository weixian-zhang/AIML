from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

try:

    # Get project client
    project_endpoint = "https://ai102-ai-foundry-aiservices.services.ai.azure.com/api/projects/ai102-ai-foundry-aiservices"
    project_client = AIProjectClient(            
            credential=DefaultAzureCredential(),
            endpoint=project_endpoint,
        )
    
    ## List all connections in the project
    connections = project_client.connections
    print("List all connections:")
    for connection in connections.list():
        print(f"{connection.name} ({connection.type})")

    print('list connection completed')

except Exception as ex:
    print(ex)