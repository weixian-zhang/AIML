from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, chain, RunnableBranch
from dotenv import load_dotenv
load_dotenv()




llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    temperature=0.0
                )


write = ChatPromptTemplate.from_template('Write a concise and short sentence about a topic: {topic}')

output_parser = StrOutputParser()

# langchain expression language, LCEL
simple_chain = write | llm | output_parser

response = simple_chain.invoke({"topic": "vampirism in real life or historically"})

extend_chain_1 = simple_chain | (lambda chain_output: 'This is really interesting: ' + chain_output)

#print(extend_chain_1.invoke({"topic": "vampirism in real life or historically"}))


# python func as runnables

@chain
def text_length(text: str) -> int:
    return len(text)

def first_half(text: str) -> str:
    print('inside first half')
    return text[: len(text) // 2]

first_half_runnable = RunnableLambda(first_half)


@chain
def random_text(text: str) -> str:
    import random
    import string

    return ''.join(random.choices(string.ascii_letters + string.digits, k=50))





# extends chain
# extend_chain_2 = extend_chain_1 |  first_half_runnable | text_length
#print(extend_chain_2.invoke({"topic": "vampirism in real life or historically"}))

# branch chain
# branch = RunnableBranch(
#     (lambda chain_output: len(chain_output) > 200, first_half | text_length),
#     text_length
# )
# branch_chain = extend_chain_1 | branch
# print(branch_chain.invoke({"topic": "vampirism in real life or historically"}))


# combine 2 chains
fact_check_chain = ChatPromptTemplate.from_template('Fact check this statement: {statement}') | llm | output_parser

# branch can output ramdom text or first-half of the chain output
branch_shorten_or_not = RunnableBranch(
    (lambda chain_output: len(chain_output) > 80, random_text),
    lambda chain_output: chain_output
)
branch_chain = extend_chain_1 | branch_shorten_or_not

dual_chain = {'statement': branch_chain} | fact_check_chain

print(dual_chain.invoke({"topic": "vampirism in real life or historically"}))