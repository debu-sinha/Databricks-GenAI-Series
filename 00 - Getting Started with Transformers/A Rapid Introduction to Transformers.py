# Databricks notebook source
dbutils.widgets.text("catalog_name","main")

# COMMAND ----------

# MAGIC %run ./init/config $catalog_name=$catalog_name

# COMMAND ----------

# MAGIC %md
# MAGIC ![DB + HF](./images/LangChain_Logo-1.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # LLMs with Hugging Face
# MAGIC In this notebook, we'll take a whirlwind tour of some top applications using Large Language Models (LLMs), as well as several key aspects of the Hugginface Transformers Library. These will include:
# MAGIC * Transformers Models
# MAGIC * Tokenizers
# MAGIC * Pipelines
# MAGIC * Datasets
# MAGIC
# MAGIC Additionally, we log our Transformers pipelines to MLflow see how Databricks MLflow integrates with LLM models out-of-the-box.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC 1. Understand how to load tokenizers and models from Huggingface using the `transformers` library
# MAGIC 2. Understand how load datasets from Huggingface using the `datasets` library
# MAGIC 3. Log our Transformers pipeline to MLflow with the `mlflow.transformers` library

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Transformers Library
# MAGIC - Let's navigate to [Huggingface model hub](https://huggingface.co/models) and explore some models that are on the platform
# MAGIC
# MAGIC ## Transformers Models and Tokenizers
# MAGIC - TODO

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging

# Load model from transformers
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# create a transformers pipeline
pipe = pipeline(
  "summarization", model=model, tokenizer=tokenizer, max_new_tokens=1024, device_map='auto', truncation=True
)

logging.set_verbosity(40)

# COMMAND ----------

text_to_summarize= """
                    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
                    """

pipe(text_to_summarize)

# COMMAND ----------

# MAGIC %md
# MAGIC # Datasets

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)

#select first 10 examples of xsum
xsum_sample = xsum_dataset["train"].select(range(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Batch Score Some Records

# COMMAND ----------

display(xsum_sample.to_pandas())

# COMMAND ----------

import pandas as pd
batch_classification_results = pipe(xsum_sample["document"], num_beams=10)

joined_data = pd.DataFrame.from_dict(batch_classification_results)\
    .rename({"summary_test": "model_summary"}, axis=1)\
    .join(pd.DataFrame.from_dict(xsum_sample))

# COMMAND ----------

display(joined_data[["document", "summary_text", "summary"]])

# COMMAND ----------

# MAGIC %md
# MAGIC # Logging and Registering with MLflow

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output

data = text_to_summarize
output = generate_signature_output(pipe, data)
signature = infer_signature(data, output)
model_name = "jpeg-mafia"

mlflow.transformers.log_model(pipe, model_name, signature=signature, input_example=data)

# COMMAND ----------

import mlflow
catalog = dbutils.widgets.get("catalog_name")
schema = schema_name
model_name = "jpeg_mafia"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri="dbfs:/databricks/mlflow-tracking/05739e2a092c4d60b3e03bc138b09e07/bd85c2dec62947858caffcabc862ab98/artifacts/mpt-7b-doan-demo",
    name=f"{catalog}.{schema}.{model_name}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View Our Model in Unity Catalog
# MAGIC
# MAGIC Now we can navigate to our `Catalog` and view our model
