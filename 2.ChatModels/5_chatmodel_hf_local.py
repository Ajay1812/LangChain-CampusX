from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "Qwen/Qwen3-0.6B"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    # device=-1,  # CPU
    pipeline_kwargs=dict(
        temperature=0.6,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm, model_id=model_id)  # ✅ pass model_id here
result = model.invoke("What is the capital of India?")
print(result.content)
