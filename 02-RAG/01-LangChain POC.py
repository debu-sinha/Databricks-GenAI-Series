# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Powering LLM Chatbot RAG PoC on Databricks
# MAGIC
# MAGIC **Authored by Debu Sinha as Part of the Gen AI Workshop Enablement Session**
# MAGIC <img style="float: right" width="600px" src="https://raw.githubusercontent.com/debu-sinha/Databricks-GenAI-Series/main/02-RAG/images/llm-genai-workshop-poc-full.jpeg">
# MAGIC
# MAGIC Welcome to the first installment of a comprehensive series designed to walk you through the entire process of creating a chatbot using Retrieval Augmentation Generation (RAG), tailored for a custom dataset. 
# MAGIC
# MAGIC In this notebook, you'll embark on a journey from downloading and cleaning a raw dataset to generating vector embeddings, culminating in their application to enhance your chatbot's contextual understanding.
# MAGIC
# MAGIC This practical exercise mirrors the initial steps typically undertaken in new projects within your organization. It aims to quickly ascertain the viability of such projects while providing a thorough understanding of the intricacies involved in deploying a full-fledged chatbot. 
# MAGIC
# MAGIC Subsequent notebooks in this series will delve into transitioning this Proof of Concept (PoC) to a production-ready deployment and explore diverse applications of Large Language Models (LLMs) using Databricks.
# MAGIC
# MAGIC
# MAGIC The notebook is divided into two primary segments:
# MAGIC
# MAGIC 1. **Data Preparation**: Here, we will ingest and refine our Databricks knowledge docs from Databricks documentation, chunk the documents, transform them into a series of embeddings stored within a vector database.
# MAGIC 2. **Q&A Inference**: This segment involves utilizing the MPT-7B Chat model to respond to queries. We will enhance the chatbot's responses by incorporating our Datbabricks documents dataset as additional context, a technique often referred to as Prompt Engineering.
# MAGIC
# MAGIC ####This notebook has been tested on 13.3 ML Runtine with GPU, Use single node g5.8xlarge machine with a single A10.
# MAGIC
# MAGIC
# MAGIC ####NOTE: Change the `catalog_name` and `schema_name` in the `util` file to point to catalog and schema where students have permision to create and read Delta tables.
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U lxml==4.9.3 langchain==0.0.335 transformers==4.35.1 accelerate==0.24.1 chromadb==0.4.17 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./util

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Section 1: Crafting and Integrating a Knowledge Base into Chroma
# MAGIC
# MAGIC <img style="float: right" width="600px" src="https://raw.githubusercontent.com/debu-sinha/Databricks-GenAI-Series/main/02-RAG/images/llm-genai-workshop-dataprep.jpg">
# MAGIC
# MAGIC In this section, we focus on creating a chatbot based on the MPT-7B Chat model. As a Databricks Feature Expert, our objective is to integrate a responsive bot into our application. This bot is designed to field customer inquiries and provide insightful recommendations on optimizing Databricks usage.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1/ Extracting Databricks documentation sitemap and pages
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/debu-sinha/Databricks-GenAI-Series/main/02-RAG/images/llm-genai-workshop-dataprep1.jpg?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC First, let's create our raw dataset as a Delta Lake table.
# MAGIC
# MAGIC For this demo, we will directly download a few documentation pages from `docs.databricks.com` and save the HTML content.
# MAGIC
# MAGIC Here are the main steps:
# MAGIC
# MAGIC - Run a quick script to extract the page URLs from the `sitemap.xml` file
# MAGIC - Download the web pages
# MAGIC - Use BeautifulSoup to extract the ArticleBody
# MAGIC - Save the result in a Delta Lake table

# COMMAND ----------

setup_datasets(dataset_type=DatasetType.DATABRICKS, reset=True)

# COMMAND ----------

display(spark.table(f"databricks_documentation_raw_{table_suffix}"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 2/ Splitting documentation pages into small chunks
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/debu-sinha/Databricks-GenAI-Series/main/02-RAG/images/llm-genai-workshop-dataprep2.jpg?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC LLM models typically have a maximum input context length, and you won't be able to compute embbeddings for very long texts.
# MAGIC In addition, the longer your context length is, the longer inference will take.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC
# MAGIC - Split document in small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends of your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one.
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference.
# MAGIC - Create multiple agents to evaluate in parallel each bigger document, and ask a final agent to craft your answer...
# MAGIC
# MAGIC ### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count how many tokens will be created for a given sequence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer and context size limit you'll be using here matches your embedding model. To do so, we'll be using the `transformers` library to count llama2 tokens with its tokenizer.

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Define maximum chunk size and the model for embedding tokenizer
max_chunk_size = 400
EMBEDDING_TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER_MODEL)
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

def split_html_on_h2(html, min_chunk_size=20, max_chunk_size=400):
    """
    Splits an HTML document into chunks based on H2 headers.

    This function ensures that the chunks are not too small by merging small chunks
    together, and also prevents chunks from being too large by splitting them 
    according to the specified maximum size. 

    Parameters:
    - html (str): HTML content to be split.
    - min_chunk_size (int, optional): Minimum size for a chunk. Defaults to 20.
    - max_chunk_size (int, optional): Maximum size for a chunk. Defaults to 400.

    Returns:
    - list: A list of text chunks.
    """
    h2_chunks = html_splitter.split_text(html)
    chunks = []
    previous_chunk = ""

    for c in h2_chunks:
        content = c.metadata.get('header2', "") + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
            previous_chunk += content + "\n"
        else:
            chunks.extend(text_splitter.split_text(previous_chunk.strip()))
            previous_chunk = content + "\n"

    if previous_chunk:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))

    return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

#lets test this method
html = spark.table(f"databricks_documentation_raw_{table_suffix}").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

from pyspark.sql.functions import *
import pandas as pd
# Let's create a user-defined function (UDF) to chunk documents
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)

# COMMAND ----------


(spark.table(f"databricks_documentation_raw_{table_prefix}")
    .withColumn('content', explode(parse_and_split('text')))
    .drop("text")
    .write.mode('overwrite').saveAsTable(f"{catalog_name}.{schema_name}.databricks_documentation_{table_suffix}"))

# COMMAND ----------

display(spark.table(f"{catalog_name}.{schema_name}.databricks_documentation_{table_suffix}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3/ Load our model to transform our docs to embeddings
# MAGIC We will simply load a sentence to an embedding model from Hugging Face and use it later in the Chroma db client.

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings

# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4/ Index the documents (rows) in our vector database
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/debu-sinha/Databricks-GenAI-Series/main/02-RAG/images/llm-genai-workshop-dataprep3.jpg?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. <br>
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Collect the text chunks with Spark; `langchain` also supports reading chunks directly from Word docs, GDrive, PDFs, etc.
# MAGIC - Create a simple in-memory Chroma vector db for storage
# MAGIC - Instantiate an embedding function from `sentence-transformers`
# MAGIC - Populate the database and save it

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.document_loaders import PySparkDataFrameLoader

#For first time turn this False
USE_CACHE=False
persist_directory="/tmp/chroma"

vector_db=None
if USE_CACHE:
  vector_db = Chroma(collection_name="databricks_documents", embedding_function=hf_embed, persist_directory=persist_directory)
else:
  reset_vector_db(persist_directory)
  loader = PySparkDataFrameLoader(spark, spark.table(f"{catalog_name}.{schema_name}.databricks_documentation_{table_suffix}"), page_content_column="content")
  split_docs=loader.load()
  vector_db = Chroma.from_documents(collection_name="databricks_documents", documents=split_docs, embedding=hf_embed, persist_directory=persist_directory)
  vector_db.similarity_search("spark") 
  vector_db.persist()

# COMMAND ----------

#if this code returns no result change USE_CACHE=False in the cell above and execute again
vector_db.similarity_search_with_relevance_scores("why should i use delta live tables", k=2) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Using the Chain for Simple Question Answering
# MAGIC That's it! It's ready to go. 

# COMMAND ----------

# Make sure you reset the GPU to free our gpu memory if you're using multiple notebooks0
# (load the model only once in 1 single notebook to avoid OOM)
reset_gpu()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Prompt engineering with langchain
# MAGIC Now we can compose with a language model and prompting strategy to make a `langchain` chain that answers questions.

# COMMAND ----------

from IPython.display import display, Markdown
from langchain import PromptTemplate


# Defining the prompt content
template_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Instruction: 
You are a Databricks Expert and your job is to help providing the best Databricks features related answers. 
Use only information in the following paragraphs to answer the question at the end. 
Explain the answer with url to source. 
Dont repeat your self and write complete sentences.

Truncate everything in your answer that starts with any non alphabetic character or numeric character. 

If you don't have any information passed to use in the paragraphs, say that you do not know.
{context}

Question: {question}

Response:
"""

prompt = PromptTemplate(input_variables=['context', 'question'], template=template_text)

#Building the chain will load MPT-7B-chat and can take several minutes
qa_chain = build_qa_chain(model_name="mosaicml/mpt-7b-chat", prompt_template=prompt, verbose=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Using the Chain for Simple Question Answering
# MAGIC That's it! It's ready to go. Try asking a Databricks related question!

# COMMAND ----------

question="why should I use Delta Live Tables for ETL?"
result = qa_chain({"input_documents": vector_db.similarity_search(query=question, k=2), "question": question})
format_and_display_chat_response(question, result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup

# COMMAND ----------

spark.sql(f"drop table {catalog_name}.{schema_name}.databricks_documentation_{table_suffix}")

# COMMAND ----------


