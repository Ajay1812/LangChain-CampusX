from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

parser = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

with open("feedback_text.txt") as f:
    feedback = f.read()

# result = classifier_chain.invoke({"feedback": feedback}).sentiment
# print(result)
# print(result.model_dump()["sentiment"])

prompt2 = PromptTemplate(
    template="write an appropiate response to this positive feedback and response should be short and crisp \n {feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template="write an appropiate response to this negative feedback and response should be short and crisp \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": feedback})
print(result)

# chain.get_graph().print_ascii() # check graph representation of your chain