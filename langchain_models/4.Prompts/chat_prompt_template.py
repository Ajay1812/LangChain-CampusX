from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain this {topic} in simple terms")
])

prompt = chat_template.invoke({"domain": "cricket", 'topic': "Dusra"})
print(prompt)