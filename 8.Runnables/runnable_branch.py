from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Write an detailed report on following {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following report {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    (RunnablePassthrough()),
    
)

final_chain = RunnableSequence(report_chain, branch_chain)
result = final_chain.invoke({"topic": "how cost function calculate in spark?"})
print(result)