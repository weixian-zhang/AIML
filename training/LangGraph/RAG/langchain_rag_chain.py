from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# STEP 1: PREPARE DATA (Documents)
# ==========================================

# Sample documents
documents = [
    "Vampires are mythological creatures that drink blood. They are often depicted as immortal beings.",
    "Historically, vampire legends arose from misunderstandings of decomposition and diseases like porphyria.",
    "Bram Stoker's Dracula, published in 1897, is one of the most famous vampire stories.",
    "Real historical figures like Vlad the Impaler inspired vampire legends.",
    "Modern vampire fiction includes works like Interview with the Vampire and Twilight."
]

# Convert to Document objects
from langchain.schema import Document
docs = [Document(page_content=doc) for doc in documents]

# ==========================================
# STEP 2: SPLIT & EMBED (Vector Store)
# ==========================================

# Create embeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# STEP 3: CREATE RAG CHAIN
# ==========================================

# LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model="gpt-4o",
    api_version="2024-12-01-preview",
    temperature=0.0
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:
""")

# Output parser
output_parser = StrOutputParser()

# ==========================================
# STEP 4: BUILD THE CHAIN
# ==========================================

# Simple RAG chain: Question → Retrieve → Format → LLM → Answer
rag_chain = (
    {
        "context": retriever,  # Retrieves relevant docs
        "question": RunnablePassthrough()  # Passes question through
    }
    | prompt  # Formats prompt with context + question
    | llm     # Generates answer
    | output_parser  # Parses output to string
)

# ==========================================
# STEP 5: USE THE CHAIN
# ==========================================

# Ask questions
question1 = "What inspired vampire legends?"
answer1 = rag_chain.invoke(question1)
print(f"Q: {question1}")
print(f"A: {answer1}\n")

question2 = "Who wrote Dracula?"
answer2 = rag_chain.invoke(question2)
print(f"Q: {question2}")
print(f"A: {answer2}\n")

question3 = "What are vampires?"
answer3 = rag_chain.invoke(question3)
print(f"Q: {question3}")
print(f"A: {answer3}")