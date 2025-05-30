from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",
                               api_key=os.environ["GEMINI_API_KEY"],
                               temperature=0.7,
                               max_completion_tokens=10)

st.header("Research tool 🔨")
paper_input = st.selectbox("Select Research Paper Name",
                           ["Attention Is All You Need",
                            "BERT: Pre-training of Deep Bidirectional Transformers",
                            "GPT-3: Language Models are Few-shot Learners",
                            "Diffusion Models Ban GANs on Image, Synthesis"])

style_input = st.selectbox("Select Explanation Style",
                           ["Beginner-Friendly",
                            "Technical",
                            "Code-Oriented",
                            "Mathematical"])

length_input = st.selectbox("Select Explanation Length",
                            ["Short (1-2 paragraphs)",
                             "Medium (3-5 paragraphs)",
                             "Long (Detailed Explanation)"])

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}  
    Explanation Length: {length_input}  
    1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable. ]
    2. Analogies:   
        - Use relatable analogies to simplify complex ideas.  
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing. 
    Ensure the summary is clear, accurate, and aligned with the provided style and length.\n,
    """,
    input_variables=["paper_input","style_input","length_input"]
)

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("submit"):
    response = model.invoke(prompt)
    st.write(response.content)
# result = model.invoke("suggest me few places to visit in Mathura?")
# print(result.content)
