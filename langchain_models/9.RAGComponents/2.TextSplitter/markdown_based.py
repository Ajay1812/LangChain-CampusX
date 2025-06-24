from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
---
### What is LangChain?

LangChain is an Open-source framework for developing applications powered by Large Language Models (LLMs).

### Why do we need LangChain?

![[Introduction to LangChain-1747658666708.jpeg]]

**Semantic-Search** -> search on the basis of its meaning.
**embedding** -> *vector* -> set of the numbers

![[Introduction to LangChain-1747659050001.jpeg]]

### Architecture Diagram

![[Introduction to LangChain-1747659491156.jpeg]]

Brain (LLM) -> NLU (Natural language understanding ) + Context Awareness Text Generation

### Advantages of LangChain:

- Concept of Chains
- Model Agnostic Development (can use any model)
- Complete Ecosystem
- Memory state handling (in conversation memory)

### Use Cases:

- Conversational Chatbots
- AI knowledge assistants
- AI Agents -> Chatbot on Steroids (not only communicate it can do work for you ex-> ticket booking from MakeMyTrip )
- Workflow Automation
- Summarization / Research helper tools 

### Alternatives of LangChain:

- *LlamaIndex*
- *HeyStack*
#### References : 

[[Fundamentals of Generative AI using LangChain]]

[[Generative AI - CampusX]]
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0
)

result = splitter.split_text(text=text)
print(result[0])
print(len(result))