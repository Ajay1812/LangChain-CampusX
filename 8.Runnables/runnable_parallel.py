from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

tweet_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

linkedin_llm = ChatHuggingFace(llm=endpoint)

prompt_tweet = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt_linkedin = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet" : RunnableSequence(prompt_tweet, tweet_llm, parser),
    "linkedin" : RunnableSequence(prompt_linkedin, linkedin_llm, parser)
})

result = parallel_chain.invoke({'topic': "AI"})

print("Tweet: ", result['tweet'])
print("LinkedIn post: ", result['linkedin'])