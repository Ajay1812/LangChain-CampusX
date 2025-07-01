from langchain_community.document_loaders import CSVLoader
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
    template="Answer the following question \n {question} from the following text \n {text}",
    input_variables=["question", "text"]
)
parser = StrOutputParser()
loader = CSVLoader(file_path="Car_sales.csv",encoding="utf-8")

docs = loader.load()

doc_list = []
for document in docs:
    doc_list.append(document.page_content)

chain = prompt | model | parser
result = chain.invoke({"question": "Average sales of Nissan cars?", "text": doc_list})
print(result)