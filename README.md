# Databricks-GenAI-Series
All the resources related to GenAI hands on workshop. To use this code clone this repo in a Databaricks workspace.

## Overview
This repository explores **Transformers, Prompt Engineering, and Retrieval-Augmented Generation (RAG)** using Apache Spark and Databricks notebooks. The project covers:

1. **Getting Started with Transformers**
   - Introduction to Hugging Face Transformers
   - Tokenization and Pre-trained Models
   - Pipelines and MLflow Integration

2. **Prompt Engineering**
   - Crafting Effective Prompts for LLMs
   - Zero-shot, Few-shot, and Chain-of-Thought Prompting
   - Using LangChain for Structured Prompts
   - Logging and Registering Prompt Chains with MLflow

3. **Retrieval-Augmented Generation (RAG) with LangChain**
   - Creating a Custom Knowledge Base from Databricks Documentation
   - Chunking and Embedding Documents
   - Indexing Data with ChromaDB
   - Querying Data with LLMs for Q&A

## Repository Structure
```
./
├── 00 - Getting Started with Transformers
│   ├── init
│   │   └── config.py
│   └── A Rapid Introduction to Transformers.py
├── 01 - Prompt Engineering
│   ├── init
│   │   └── config.py
│   └── Prompt Engineering.py
└── 02-RAG
    ├── 01-LangChain POC.py
    ├── config.py
    └── util.py
```

## Installation and Setup
### Install Dependencies
```sh
pip install -U lxml langchain transformers accelerate chromadb mlflow
```

### Configuration
Set the required database catalog and schema in `config.py` before running the notebooks.

## 1️⃣ Getting Started with Transformers

### Load and Use Pre-Trained Models
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
```

### Load Dataset and Summarize Text
```python
from datasets import load_dataset
xsum_dataset = load_dataset("xsum", version="1.2.0")
xsum_sample = xsum_dataset["train"].select(range(10))
batch_results = pipe(xsum_sample["document"], num_beams=10)
```

### Log and Register Model with MLflow
```python
import mlflow
from mlflow.models import infer_signature

signature = infer_signature("input_example", batch_results)
mlflow.set_experiment("/Users/demo/genai-intro-workshop")

with mlflow.start_run():
    mlflow.transformers.log_model(pipe, "pegasus-summarizer", signature=signature)
```

## 2️⃣ Prompt Engineering

### Using LangChain for Prompt Templates
```python
from langchain import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a Databricks support engineer.
    Include relevant details in your response.
    User Question: {question}
    """
)
```

### Zero-shot vs Few-shot Prompting
```python
zero_shot_template = """Describe sentiment of the tweet:
[Tweet]: {input_string}
"""

few_shot_template = """Describe sentiment of the tweet:
[Tweet]: "I hate it when my phone battery dies." [Sentiment]: Negative
[Tweet]: "My day has been 👍" [Sentiment]: Positive
[Tweet]: {input_string} [Sentiment]:
"""
```

### Chain-of-Thought Prompting
```python
chain_of_reasoning_prompt = """
For the following question, explain your reasoning step by step:
{input_string}
"""
```

### No-Hallucination Prompting
```python
no_hallucinations_prompt = """
Only respond if you have sufficient information.
Otherwise, say: "Sorry, I don't have enough information."
Question: {input_string}
"""
```

### Log LangChain Model with MLflow
```python
import mlflow
mlflow.langchain.log_model(llama_chain, "prompt-engineering-llm")
```

## 3️⃣ Retrieval-Augmented Generation (RAG)

### Chunking and Embedding Text
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

max_chunk_size = 400
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size)
```

### Creating a Vector Database with Chroma
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma.from_documents(documents=split_docs, embedding=hf_embed, persist_directory="/tmp/chroma")
```

### Querying Documents using an LLM Chain
```python
from langchain.chains.question_answering import load_qa_chain

template = """Use the following context to answer the question:
{context}
Question: {question}
"""

qa_chain = load_qa_chain(llm=llama_model, chain_type="stuff", prompt=PromptTemplate(input_variables=["context", "question"], template=template))
question="What are Delta Live Tables?"
result = qa_chain({"input_documents": vector_db.similarity_search(question), "question": question})
print(result)
```

## License
This repository is open for educational purposes. Feel free to modify and improve the content.
