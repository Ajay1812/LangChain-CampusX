from langchain_community.document_loaders import WebBaseLoader
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

url = "https://www.flipkart.com/apple-macbook-air-m4-16-gb-512-gb-ssd-macos-sequoia-mw103hn-a/p/itmc2cfaee6a7b6e?pid=COMH9ZWQKQ5D3ZED&lid=LSTCOMH9ZWQKQ5D3ZEDKELRCZ&marketplace=FLIPKART&q=macbook+air+m4&store=6bo%2Fb5g&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_7_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_7_na_na_na&fm=organic&iid=cbde1b09-c2cd-4487-b044-7d2eda765c9f.COMH9ZWQKQ5D3ZED.SEARCH&ppt=None&ppn=None&ssid=01m0acfzkg0000001750676035619&qH=a3dc101ea3bce06d"
loader = WebBaseLoader(web_path=url)

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({"question": "what is the peak brightness of product?", "text": docs[0].page_content})
print(result)