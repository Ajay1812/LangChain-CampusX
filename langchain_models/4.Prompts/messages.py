from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",
                               api_key=os.environ["GEMINI_API_KEY"],
                               temperature=0.6)

messages = [
    SystemMessage(content="You are a helpful assitant"),
    HumanMessage(content="Tell me about LangChain")
] 

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)