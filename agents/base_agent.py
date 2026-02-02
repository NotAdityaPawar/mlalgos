from langchain_aws import ChatBedrock
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


load_dotenv()




class BaseAgent:
    def __init__(self, llm: ChatBedrock, tools= None, memory = None):
        self.llm = llm
        self.tools = tools or {}
        self.memory = memory or []
                
    def observe(self, observation):
        self.memory.append({"role": "observation", "content": observation})
        
    def think(self, prompt):
        raise NotImplemented
    
    def act(self, action):
        raise NotImplemented
    
    def run(self, task):
        return self.llm.invoke(HumanMessage(content = task))
    
if __name__ == "__main__":
    
    llm = ChatBedrock(
        model="us.amazon.nova-pro-v1:0",
        aws_access_key_id= os.environ.get("aws_access_key_id"),
        aws_secret_access_key= os.environ.get("aws_secret_access_key"),
        aws_session_token= os.environ.get("aws_session_token")
    )
    agent = BaseAgent(
        llm = llm,    
    )
    
    agent.run("what is 2 + 2x   ")
    
    
