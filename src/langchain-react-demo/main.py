from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

load_dotenv()


@tool
def search(query: str):
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Query: {query}")
    return tavily.search(query)
    
    
def main():
    template = "find me the top 3 jobs for this role: Java engineer"
    
    prompt_template = PromptTemplate(input_variables=["job role"], template=template)
    
    tools = [search]
    llm = ChatOllama(temperature=0.0, model="gemma4:e2b")
    agent = create_agent(model=llm, tools=tools)
    
    chain = prompt_template | llm
    
    response = agent.invoke({"messages":HumanMessage(content=template)})
    print(response)
    

if __name__ == "__main__":
    main()
