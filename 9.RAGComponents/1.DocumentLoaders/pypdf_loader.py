from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    api_key=os.environ["GEMINI_API_KEY"]
)

prompt = PromptTemplate(
    template="What is docker in the following text - \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

loader = PyPDFLoader(
    file_path="docker.pdf",
)

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})
print(result)


