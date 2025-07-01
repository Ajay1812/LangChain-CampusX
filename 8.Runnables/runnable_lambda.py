from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke in short - {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model, parser),
    'word_count' : RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen, parallel_chain)
result = final_chain.invoke({'topic': "Hello world!"})

final_result = """joke - {} \nExplaination - {} \nword count - {}""".format(result['joke'], result['explaination'], result['word_count'])

print(final_result)