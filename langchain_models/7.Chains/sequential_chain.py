from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on the {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointers summary from the following text \n {text}",
    input_variables=["text"]
)
parser = StrOutputParser()

chain = (
    prompt1
    | model 
    | parser 
    | prompt2
    | model 
    | parser
)


result = chain.invoke({"topic": "Python for everybody"})
print(result)

# chain.get_graph().print_ascii()