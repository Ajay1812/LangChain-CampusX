from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",
                               api_key=os.environ["GEMINI_API_KEY"],
                               temperature=0.7)

# prompt: Detailed Report 
template1 = PromptTemplate(
    template="Write a Detailed Report on {topic}",
    input_variables=["topic"]
)

# prompt: summarize report in 5 lines
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic': 'Black hole'})

result = model.invoke(prompt1)
# print(result)
prompt2 = template2.invoke({'text': result.content})

result1 = model.invoke(prompt2)
print(result1.content)