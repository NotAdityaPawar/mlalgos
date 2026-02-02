from langchain_aws import ChatBedrock
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


load_dotenv()


# defining the Model with access tokens
llm = ChatBedrock(
    model="us.amazon.nova-pro-v1:0",
    aws_access_key_id= os.environ.get("aws_access_key_id"),
    aws_secret_access_key= os.environ.get("aws_secret_access_key"),
    aws_session_token= os.environ.get("aws_session_token")
)


# custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Dummy implementation for illustration purposes
    return f"The current weather in {city} is sunny with a temperature of 75°F."


llm_with_tool = llm.bind_tools([get_weather])

if __name__ == "__main__":
    
    response = llm_with_tool.invoke([HumanMessage(content = "What's the weather in SF?")])
    print(response.tool_calls)
    print(response)