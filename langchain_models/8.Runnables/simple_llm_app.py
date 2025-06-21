from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

prompt = PromptTemplate(
    template="write a cachy one title for blog about {topic}",
    input_variables=["topic"]
)

topic = input("Enter your topic: ")
formatted_prompt = prompt.format(topic=topic)

blog_title = model.predict(formatted_prompt) # .predict (deprecated) use invoke
print(f"Generating blog title: ", blog_title)