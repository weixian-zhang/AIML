import os
from openai import AzureOpenAI
import time
from datetime import datetime, timezone
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError

# assistant samples
# https://github.com/Azure-Samples/azureai-samples/blob/main/scenarios/Assistants/api-in-a-box/math_tutor/assistant-math_tutor.ipynb

# quickstart
# https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-quickstart?tabs=command-line&pivots=programming-language-python

# function call example:
# https://github.com/gbaeke/azure-assistants-api/blob/main/files.ipynb

instructon = '''
You are helpful assistant known as Kusto-WingMan with the knowledge of Azure Kusto query language.
Your job is to translate questions in natural language to Kusto or KQL query

Kustoers ask question about what do they want to find out about web requests, Azure resources or Azure Monitor,
you will help to generate relevant Kusto query

Rules:

1. Always address the user as kustoer
'''

openai_assistant_name = 'Kusto Wingman'
openai_assistant = None
openai_endpoint = 'https://openai-assistant-austeast-1.openai.azure.com/openai'
openai_api_key = os.environ.get('OPENAI_KEY')
openai_deployment_id = 'kusto_wingman_gpt_4'
openai_base_url=f"{openai_endpoint}" #/openai/deployments/{openai_deployment_id}/extensions"
openai = AzureOpenAI(api_key=openai_api_key, 
                     base_url=openai_endpoint,
                     api_version="2024-02-15-preview")
openai_threadid = ''

curr_dir = os.path.dirname(os.path.realpath(__file__))
app_insights_knowledge_file = os.path.join(curr_dir, 'knowledge', 'app_insights_knowledge.json')

# done in portal
def upload_app_insights_file():
    files = openai.files.list()

    for f in files:
        if f.filename == 'app_insights_knowledge.json':
            return f.id
        
    return openai.files.create(
        file=open(app_insights_knowledge_file, 'rb'), purpose='assistants')
    
# wait for retrieval assistant to be available, TODO
def create_assistant():

    app_insights_knowledge_file = upload_app_insights_file()

    asistants = openai.beta.assistants.list()

    for a in asistants:
        if a.name == openai_assistant_name:
            return a
        
    assistant = openai.beta.assistants.create(
        name=openai_assistant_name,
        instructions=instructon,
        model='gpt-4-1106-preview',
        file_ids = [app_insights_knowledge_file],
        tools=[ 
                {
                "type": "code_interpreter",  # should be set to retrieval but that is not supported yet; required or file_ids will throw error
                },
                { 
                "function": {
                    "name": "run_app_insights_query",
                    "description": "run application insights query",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "application insights query"
                            }
                    },
                    "required": [
                        "employee",
                        "amount"
                    ]
                    }
                },
                "type": "function"
                }
            ],
    )
    return assistant

def wait_for_run(run, thread_id):
    while run.status == 'queued' or run.status == 'in_progress':
        run = openai.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
        )
        time.sleep(0.5)

    return run


# function to execute as instructed from assistant run result
def run_app_insights_query(query):
    pass

def run_resource_graph_query(query):
    pass

def run_az_monitor_query(query):
    pass


def init():
    # Create a thread
    thread = openai.beta.threads.create()
    
    assistant = create_assistant()

    return thread.id, assistant.id


def create_message(threadId:str, assistantId: str, content: str):

    messages = []
    queryResult = []

    # creates user message
    user_message = openai.beta.threads.messages.create(
        thread_id=threadId,
        role='user',
        content=content
    )

    run = openai.beta.threads.runs.create(
        thread_id= threadId,
        assistant_id= assistantId
    )

    run = wait_for_run(run, threadId)

    if run.required_action:
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        for tc in tool_calls:
            func_name = tc.function.name
            query_arg = tc.function.arguments

            if func_name == 'run_app_insights_query':
                
                queryResult = run_app_insights_query(query_arg)
            
    
    messageList = openai.beta.threads.messages.list(thread_id=threadId)
    # for m in messageList.data:
    #     messages.append((m.role, m.content[0].text.value))
    
    return messageList.data[0].content[0].text.value, queryResult

        
#threadId, assistantId = init()

#result = create_message(threadId, assistantId, 'show me the total requests that is more than 1 day ago summarize by duration in percentage')
