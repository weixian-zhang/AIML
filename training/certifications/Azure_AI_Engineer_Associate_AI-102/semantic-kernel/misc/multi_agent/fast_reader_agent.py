import os
from dotenv import load_dotenv, find_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AgentGroupChat
from semantic_kernel import Kernel
from semantic_kernel.agents.strategies import TerminationStrategy, SelectionStrategy, KernelFunctionSelectionStrategy
from semantic_kernel.functions import kernel_function, KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistoryTruncationReducer
import asyncio

load_dotenv(find_dotenv())

project_client = AzureAIAgent.create_client(
        #https://ai102-ai-foundry-aiservices.services.ai.azure.com/api/projects/ai102-ai-foundry-aiservices
        endpoint='https://ai102-ai-foundry-aiservices.services.ai.azure.com/api/projects/ai102-ai-foundry-aiservices', #os.getenv("PROJECT_ENDPOINT_STRING"),
        credential=DefaultAzureCredential()
    )


# code ref
# https://github.com/MicrosoftLearning/mslearn-ai-agents/blob/main/Labfiles/05-agent-orchestration/Python/agent_chat.py
async def create_agent(name, description, instruction) -> AzureAIAgent:

    agent = AzureAIAgent(
        name=name,
        client=project_client,
        definition= await project_client.agents.create_agent(
            model='gpt-4',
            name =name,
            description=description,
            instructions= instruction
        )
    )

    return agent
    

class SummarizedTerminationStrategy(TerminationStrategy):
    """Custom termination strategy that terminates when the last agent has responded."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "__ALL_SUMMARY_END__" in history[-1].content



async def main():

    kernel = Kernel()

    fast_reader_agent_desc = '''A fast reader agent that makes use of other registered agents to get story books first page,
    and summarizes book content in a few sentences.
    This agent will return the summary as a termination agent'''
    fast_reader_agent = await create_agent(
        name="fast_reader_agent",
        description=fast_reader_agent_desc,
        instruction=fast_reader_agent_desc
    )


    story_book_agent_desc = """A story book agent that retrieves story books based on user requests. 
    "Retrieves top 3 story book first page that you think are most releveant to user's request.
    For each story book first page, each separated with __STORY_BOOK_END_. 
    the final content result of al 3 story book first page, '__ALL_STORY_BOOK_END__' to mark the end of the content."""
    story_book_agent = await create_agent(
        name="story_book_agent",
        description=story_book_agent_desc,
        instruction=story_book_agent_desc,
        
    )

    summarize_book_agent_desc = """A summarization agent that provides concise summaries of all 3 story books first paragraphs.
    Each story book first page is separated by __STORY_BOOK_END_, summarize each story book first page in 5 lines or less.
    Each summarized story book first page is separated by __SUMMARY_END_,
    The final content result of all 3 story book first page summaries, '__ALL_SUMMARY_END__' to mark the end of the content."""
    summarize_book_agent = await create_agent(
        name="summarize_book_agent",
        description=summarize_book_agent_desc,
        instruction=summarize_book_agent_desc,
    )

    selection_function = KernelFunctionFromPrompt(
                function_name="selection",
                prompt=f"""
                You are an agent selection function, examine theprovided response, if you see __ALL_STORY_BOOK_END__ in response,
                means story-book agent has completed retrieving story book and next summarizer agent should take over.
                If you do not see __ALL_STORY_BOOK_END__ means no story book is retrieved yet, so the story book agent should continue to take over.
                
                Choose only from these participants:
                - {story_book_agent}
                - {summarize_book_agent}

                Rules:
                - {story_book_agent} is the one that can retrieve top 3 story book first page and return the results whenever user asks about story books.
                - {summarize_book_agent} is the one that can summarize the first page of top 3 story books and return the results whenever user asks for summaries.

                History:
                {{$history}}
                """
            )
    
    termination_function = KernelFunctionFromPrompt(
                function_name="termination",
                prompt=f"""
                if you see __ALL_SUMMARY_END__ in response, means summarization agent has completed summarizing story book first page and the conversation can be terminated.
                If you do not see __ALL_SUMMARY_END__ means summarization agent has not completed
                """)

    kernel = Kernel()

    agc = AgentGroupChat(
        agents=[fast_reader_agent, story_book_agent, summarize_book_agent],
        termination_strategy=SummarizedTerminationStrategy(),
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=fast_reader_agent,
            agent_variable_name="agents",
            kernel=kernel,
            function=selection_function,
            result_parser=lambda result: result.content,
             history_variable_name="history",
             history_reducer=ChatHistoryTruncationReducer(target_count=5)
        ))

    response = await agc.add_chat_message("I would like to summarize top 3 Stephen King horror story book content")

    print(response)

    await project_client.agents.delete_agent(fast_reader_agent.name)
    await project_client.agents.delete_agent(story_book_agent.name)
    await project_client.agents.delete_agent(summarize_book_agent.name)


asyncio.run(main())

    


