
from openai import AzureOpenAI
import os

openai_key = os.environ['OPENAI_KEY']
openai_endpoint = os.environ['OPENAI_ENDPOINT']

print(openai_key)
print(openai_endpoint)

client = AzureOpenAI(
    api_key=openai_key,
    api_version="2023-12-01-preview",
    azure_endpoint=openai_endpoint
)

# define context
# detail description or instruction
# directing prompt style like a "director" in a movie e.g: the quizshould be challenging but not overly complex
# quantitative (measurement) e.g: for each question, give a difficulty of 7 out of 1 to 10

basic_prompt = 'python quiz, 5 questions and 4 options each'

completion = client.completions.create(
    model='gpt-35-turbo',
    prompt=basic_prompt,
    max_tokens='infinite'
)

print(completion.choices[0].text)