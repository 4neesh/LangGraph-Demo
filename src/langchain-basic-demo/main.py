from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    
    information_param = """
        There are 20 teams in the premier league.
        The best team is Leeds.
        They are based in Yorkshire.
    """
    summary_template = """
        Given the information: {information},
        Summarise the information, stating the best team and where they are from.
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template = summary_template
    )
    
    llm = ChatOllama(model="gemma4:e2b", temperate=0.0)
    # LangChain Expression Language - we pass the chain of our prompt to the LLM
    # The output of the left is turned into the input of the right
    chain = summary_prompt_template | llm
    print("invoking")
    response = chain.invoke(input={"information":information_param})
    print(response.content)
        
if __name__ == "__main__":
    main()