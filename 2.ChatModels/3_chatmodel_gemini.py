from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", 
                                api_key=os.environ["GEMINI_API_KEY"],
                                temperature=0.9, max_completion_tokens=10)
result = model.invoke("suggest me few places to visit in Mathura?")      

print(result.content)