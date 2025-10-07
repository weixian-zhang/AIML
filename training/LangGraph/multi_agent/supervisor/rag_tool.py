import os
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()
# https://blog.futuresmart.ai/langgraph-agent-with-rag-and-nl2sql#heading-agentic-rag-tool-integration

def load_document():
    doc_path = os.path.join(os.path.dirname(__file__), "./data/azure_fundamentals.pdf")
    loader = PyPDFLoader(doc_path)
    document = loader.load()
    return document


# documents = load_document()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

# chunks = text_splitter.split_documents(documents)

# collection_name = "azure_fundamentals"


# embeddings = AzureOpenAIEmbeddings()


# vectorstore = Chroma.from_documents(documents=chunks, 
#                                     collection_name=collection_name,
#                                     embedding=embeddings
#                                     )

# @tool
# def rag_tool(query: str) -> str:
#     docs = vectorstore.similarity_search(query, k=5)
#     context = "\n".join([doc.page_content for doc in docs])
#     return context

@tool(description="Use this tool to do semantic search for information from the Azure Fundamentals PDF document.")
def rag_tool(query: str) -> str:
    return "RAG tool response"