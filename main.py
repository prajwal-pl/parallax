from dotenv import load_dotenv
from core.tool import tool
from core.agent import Agent
import os

from tavily import TavilyClient

load_dotenv()

@tool
def web_search(query:str):
    """
    Search the web to fetch results and answers
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.research(query)

    print(response)
    return response

@tool
def calculate(exp:str):
    """
    Calculate the result of a math expression
    """
    return str(eval(exp))

def main():
    agent = Agent(model="cohere/north-mini-code:free", system_prompt="You are an assistant which uses tools whenever neccessary to solve a given task")

    agent.add_tool(calculate)

    response = agent.run("What is 225498 * 465422 + 13231688")

    print(f"Result: {response}")
    

if __name__ == "__main__":
    main()  
