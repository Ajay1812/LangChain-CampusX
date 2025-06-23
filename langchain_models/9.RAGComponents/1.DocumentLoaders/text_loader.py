from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash-8b",
    model="gemini-2.5-flash-lite-preview-06-17",
    api_key=os.environ["GEMINI_API_KEY"]
)
prompt = PromptTemplate(
    template="write a summary for the following song - \n {text}",
    input_variables=["text"]
)
loader = TextLoader(file_path="gukesh_song.text",encoding="utf-8")
parser = StrOutputParser()

docs = loader.load()
# print(type(docs))
# print(len(docs))
# print(type(docs[0]))

# print(docs[0].page_content)
# print(docs[0].metadata)

parallel_chain = RunnableParallel({
        "Song" : RunnablePassthrough(),
        "summary" : prompt | model | parser
})

result = parallel_chain.invoke({"text": docs[0].page_content})
print(result)