from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Rubix Cube"})
print(result)

chain.get_graph().print_ascii() # graph representation of your chain