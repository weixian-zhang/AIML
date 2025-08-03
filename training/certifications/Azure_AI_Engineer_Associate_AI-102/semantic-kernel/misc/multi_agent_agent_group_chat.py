import os
from dotenv import load_dotenv, find_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistoryTruncationReducer, ChatMessageContent, AuthorRole
from semantic_kernel.agents.strategies import (TerminationStrategy, KernelFunctionSelectionStrategy, SequentialSelectionStrategy, SelectionStrategy,
                                               KernelFunctionTerminationStrategy)
from semantic_kernel.functions import kernel_function, KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistoryTruncationReducer
import asyncio

load_dotenv(find_dotenv())

# project_client = AzureAIAgent.create_client(
#         #https://ai102-ai-foundry-aiservices.services.ai.azure.com/api/projects/ai102-ai-foundry-aiservices
#         endpoint='https://ai102-ai-foundry-aiservices.services.ai.azure.com/api/projects/ai102-ai-foundry-aiservices', #os.getenv("PROJECT_ENDPOINT_STRING"),
#         credential=DefaultAzureCredential()
#     )


# code ref
# https://github.com/MicrosoftLearning/mslearn-ai-agents/blob/main/Labfiles/05-agent-orchestration/Python/agent_chat.py
# https://learn.microsoft.com/en-us/semantic-kernel/support/archive/agent-chat?pivots=programming-language-python
# https://systenics.ai/blog/2025-04-22-understanding-selection-and-termination-strategy-functions-in-dotnet-semantic-kernel-agent-framework/

# was using AI Agent but switch to ChatCompletionAgent
# async def create_agent(name, description, instruction) -> ChatCompletionAgent:

#     agent = ChatCompletionAgent(
#         name=name,
#         client=project_client,
#         definition= await project_client.agents.create_agent(
#             model='gpt-4',
#             name =name,
#             description=description,
#             instructions= instruction
#         )
#     )

#     return agent


# custom selection strategy
class AgentSelectionStrategy(SelectionStrategy):
    """Custom selection strategy that selects the next agent based on the last agent's response."""

    async def select_agent(self, agents, history):
        """Select the next agent to interact with."""
        if not history:
            return agents[0]  # If no history, return the first agent

        last_message = history[-1]
        if "__ALL_PAPER_END__" in last_message.content:
            return agents[-1]  # If last message indicates all papers are done, return the summarization agent

        return agents[0]  # Otherwise, return the first agent
    

# class SummarizedTerminationStrategy(TerminationStrategy):
#     """Custom termination strategy that terminates when the last agent has responded."""

#     async def should_agent_terminate(self, agent, history):
#         """Check if the agent should terminate."""
#         return "__ALL_SUMMARY_END__" in history[-1].content
    




async def main():

    kernel = Kernel()
   

    chat_completion_service = AzureChatCompletion(
        service_id="default",
        deployment_name="gpt-4",
        api_version='2025-01-01-preview',
        api_key=os.getenv("openai_key"),
        endpoint=os.getenv("PROJECT_ENDPOINT_STRING")
    )

    kernel.add_service(service=chat_completion_service)


    genai_paper_agent_name = "genai_paper_agent"
    summarize_paper_agent_name = "summarize_paper_agent"


    # fast_reader_agent_desc = '''A fast reader agent that makes use of other registered agents to get story books first page,
    # and summarizes book content in a few sentences.
    # This agent will return the summary as a termination agent'''
    # fast_reader_agent = ChatCompletionAgent(
    #     name='fast_reader_agent',
    #     kernel=kernel,
    #     description= fast_reader_agent_desc,
    #     instructions= fast_reader_agent_desc,
    # )


    genai_paper_agent_desc = """A generative ai paper agent that retrieves generative ai papers based on user requests. 
    "Retrieves top 3 short generative ai papers that you think are most releveant to user's request."""
    genai_paper_agent = ChatCompletionAgent(
        name=genai_paper_agent_name,
        kernel=kernel,
        instructions= genai_paper_agent_desc,
    )

    summarize_paper_agent_desc = """A summarization agent that provides concise summary of all 3 papers.
    Each paper is summarized to 5 lines or less.
    The summary is returned as a single response ending with the word "__ALL_SUMMARY_END__"."""
    summarize_paper_agent = ChatCompletionAgent(
        name=summarize_paper_agent_name,
        kernel=kernel,
        instructions= summarize_paper_agent_desc,
    )


    # and begin to summarize each of the first page of top 3 story books returned by {genai_paper_agent_name} agent, and return the summarized results.
    #                - {genai_paper_agent_name} is the one that can retrieve top 3 story book first page and return the results ending with __ALL_PAPER_END__.
    selection_function = KernelFunctionFromPrompt(
                function_name="selection",
                prompt=f"""
                Determine which participant takes the next turn in a conversation based on the the most recent participant.
                State only the name of the participant to take the next turn, with no breakline or space.
                No participant should take more than one turn in a row.
                
                Choose only from these participants:
                - {genai_paper_agent_name}
                - {summarize_paper_agent_name}

                - {genai_paper_agent_name} will start.
                - If response contains all paper content, select {summarize_paper_agent_name} agent for next response.


                History:
                {{$lastmessage}}
                """
            )
    
    termination_function = KernelFunctionFromPrompt(
                function_name="termination",
                prompt=f"""
                
                If response contains the word "__ALL_SUMMARY_END__", reply with only one word "done"

                History
                {{$lastmessage}}
                """)

    selection_result_parser = lambda result: (
        # print(f"Selection result: {result.value}")
        # [print(f'role:{v["role"]}, content:{v["content"]}') for v in result.metadata['arguments']['lastmessage']]
        # print('\n\n')
        result.value[0].content.strip() if result and result.value else summarize_paper_agent_name
    )


    agc = AgentGroupChat(
        agents=[genai_paper_agent, summarize_paper_agent],
        
        termination_strategy= KernelFunctionTerminationStrategy(
            agents= [summarize_paper_agent],
            agent_variable_name="agents",
            kernel=kernel,
            function=termination_function,
            result_parser=lambda result: (
                True if result.value[0].content == "done" else False
            ),
            history_variable_name="lastmessage",
            #history_reducer=ChatHistoryTruncationReducer(target_count=1),
            #maximum_iterations=2
        ),

        # selection_strategy=AgentSelectionStrategy()
        
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=genai_paper_agent,
            agent_variable_name="agents",
            kernel=kernel,
            function=selection_function,
            result_parser=lambda result: (
            # print(f"Selection result: {result.value}")
            # [print(f'role:{v["role"]}, content:{v["content"]}') for v in result.metadata['arguments']['lastmessage']]
            # print('\n\n')
            result.value[0].content.strip() if result and result.value else summarize_paper_agent_name
        ),
            history_variable_name="lastmessage",
            #history_reducer=ChatHistoryTruncationReducer(target_count=1)
        )

        # selection_strategy=SequentialSelectionStrategy(
        #     agents=[genai_paper_agent, summarize_paper_agent],
        #     initial_agent=genai_paper_agent
        # )
    )


   

    # chat completion test
    # chat_history = ChatHistory(
    #     messages=[ChatMessageContent(AuthorRole.USER, content="I would like to summarize top 3 Stephen King horror story book content")]
    # )
    # response = await chat_completion_service.get_chat_message_content(
    #     chat_history=chat_history,
    #     settings= OpenAIChatPromptExecutionSettings(top_p=0.1, temperature=0.7, max_tokens=1000)
    #     )

    # print(response)
    await agc.add_chat_message(ChatMessageContent(AuthorRole.USER, content="I would like to retrieve and summarize any 3 generative AI paper that is used in real world to solve problems like in healthcare or finance or cybersecurity."))
    

    try:
        # Invoke a response from the agents
        async for response in agc.invoke():
            if response is None or not response.name:
                continue
            # print(f"{response.content}")
    except Exception as e:
        print(f"Error during chat invoke: {e}")

    # print summary only
    async def get_first_element(async_iterable):
        async for item in async_iterable:
            return item
        

    first_item = await get_first_element(agc.get_chat_messages())
    print(f"Chat History:{first_item.content}")

    # await project_client.agents.delete_agent(fast_reader_agent.name)
    # await project_client.agents.delete_agent(genai_paper_agent.name)
    # await project_client.agents.delete_agent(summarize_paper_agent.name)


asyncio.run(main())

    


