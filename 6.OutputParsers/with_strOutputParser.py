from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

# gemini-1.5-flash-8b
# gemini-2.0-flash-lite
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",
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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Black hole"})
print(result)