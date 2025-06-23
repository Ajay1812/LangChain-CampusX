from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

# model = ChatGoogleGenerativeAI(
#     # model="gemini-1.5-flash-8b",
#     model="gemini-2.5-flash-lite-preview-06-17",
#     api_key=os.environ["GEMINI_API_KEY"]
# )
# prompt = PromptTemplate(
#     template="write a summary for the following song - \n {text}",
#     input_variables=["text"]
# )

loader = DirectoryLoader(
    path="Books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()
# print(len(docs))

for document in docs:
    print(document.metadata)