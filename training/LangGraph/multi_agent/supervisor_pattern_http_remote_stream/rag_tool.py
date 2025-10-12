import os
from langchain.document_loaders import PyPDFLoader
from typing import Annotated
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from state import SupervisorState

load_dotenv()
# https://blog.futuresmart.ai/langgraph-agent-with-rag-and-nl2sql#heading-agentic-rag-tool-integration

def load_document():
    doc_path = os.path.join(os.path.dirname(__file__), "./data/azure_fundamentals.pdf")
    loader = PyPDFLoader(doc_path)
    document = loader.load()
    return document


documents = load_document()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

collection_name = "azure_fundamentals"


embeddings = AzureOpenAIEmbeddings()


vectorstore = Chroma.from_documents(documents=chunks, 
                                    collection_name=collection_name,
                                    embedding=embeddings
                                    )



@tool(description="RAG tool sto do semantic search for information from vector database to answer user query")
def rag_tool(query: str, 
             state: Annotated[SupervisorState, InjectedState], 
             tool_call_id: Annotated[str, InjectedToolCallId]) -> dict[str, str]:
    

    assert query, 'query is required'
    
    # must return ToolMessage after tool call if custom Command is return
    # if not, error:
        # Expected to have a matching ToolMessage in Command.update for tool 'rag_tool', got: []. 
        # Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage. 
        # You can fix it by modifying the tool to return `Command(update=[ToolMessage("Success", tool_call_id=tool_call_id), ...], ...)`
    tool_message = ToolMessage(
        content=f"Successfully used RAG tool to get information",
        name="rag_tool",
        tool_call_id=tool_call_id
    )

    search_result = vectorstore.similarity_search(query=query, k=3)

    result = '\n\n'.join([doc.page_content for doc in search_result])

    return  Command(update= {"messages": [tool_message], "rag_content": result})