from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",
                               api_key=os.environ["GEMINI_API_KEY"],
                               temperature=0.7,
                               max_completion_tokens=10)

st.header("Research tool 🔨")
paper_input = st.selectbox( "Select Research Paper Name",
                            ["Attention Is All You Need",
                             "BERT: Pre-training of Deep Bidirectional Transformers",
                             "Transformer Architectures and Their Applications in Time Series Forecasting",
                             "Scalable ETL Workflows Using Apache Airflow in Cloud Environments",
                             "GPT-3: Language Models are Few-Shot Learners",
                             "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style",
                            ["Beginner-Friendly",
                             "Technical",
                             "Code-Oriented",
                             "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length",
                             ["Short (1-2 paragraphs)",
                              "Medium (3-5 paragraphs)",
                              "Long (detailed explanation)"] )

template = load_prompt("template.json")

# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })

if st.button("submit"):
    chain = template | model

    result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
    })

    st.write(result.content)
# result = model.invoke("suggest me few places to visit in Mathura?")
# print(result.content)
