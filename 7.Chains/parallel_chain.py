from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate short and simple notes on the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question/answer from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

with open("transformer_text.txt") as f:
    text = f.read()

result = chain.invoke({"text": text})
print(result)

chain.get_graph().print_ascii()