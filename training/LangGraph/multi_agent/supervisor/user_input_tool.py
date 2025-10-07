from langchain.tools import tool

@tool
def user_input_tool():
    user_input = input("Enter your search: ")