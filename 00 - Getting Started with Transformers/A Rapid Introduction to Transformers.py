# Databricks notebook source
dbutils.widgets.text("catalog_name","main")

# COMMAND ----------

# MAGIC %run ./init/config $catalog_name=$catalog_name

# COMMAND ----------

# MAGIC %md
# MAGIC ![DB + HF](https://github.com/debu-sinha/Databricks-GenAI-Series/blob/main/00%20-%20Getting%20Started%20with%20Transformers/images/genai_intro_banner.png?raw=true)
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
# MAGIC ## Hugging Face, Transformers Models, and Tokenizers
# MAGIC Let's navigate to [Hugging Face model hub](https://huggingface.co/models) and explore some models that are on the platform. You will see that (most) models come with descriptions of their task, the data that they were trained on, as well as their associated `Tokenizer`
# MAGIC
# MAGIC ### Transformers Models
# MAGIC In the Hugging Face library, a Transformers model refers to a pre-trained model that can be used for a wide range of NLP tasks. These models, like BERT, GPT, or T5, are built using the Transformers architecture and are trained on large datasets, enabling them to understand and generate human language effectively.
# MAGIC
# MAGIC ### Tokenizers
# MAGIC A tokenizer is a critical component that preprocesses text data for the model. Each model in the library usually comes with its associated tokenizer, ensuring compatibility and optimal performance.

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging

# Load model from Hugging Face using the transformers library
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# create a transformers pipeline
# to create one, we need to define the following: The type of task, the model, the tokenizer, as well as a few other parameters that we'll discuss below
pipe = pipeline(
  "summarization", model=model, tokenizer=tokenizer, max_new_tokens=1024, device_map='auto', truncation=True
)

logging.set_verbosity(40)

# COMMAND ----------

text_to_summarize= """
                    Barrington DeVaughn Hendricks (born October 22, 1989), known professionally as JPEGMafia (stylized in all caps), is an American rapper, singer, and record producer born in New York City and based in Baltimore, Maryland. His 2018 album Veteran, released through Deathbomb Arc, received widespread critical acclaim and was featured on many year-end lists. It was followed by 2019's All My Heroes Are Cornballs and 2021's LP!, released to further critical acclaim. 
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
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/1*OVqzvRSNWloHMYCF1EZtqg.png" alt="mlflow" width="700"/>
# MAGIC
# MAGIC # Logging and Registering with MLflow
# MAGIC Now that we have our model, we want to log the model and its artifacts, so we can version it, deploy it, and also share it with other users.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Model Signature
# MAGIC For LLMs, we need to generate a [model signature](https://mlflow.org/docs/latest/models.html#model-signature-and-input-example).
# MAGIC Model signatures show the expected input and output types for a model. Which makes quality assurance for downstream serving easier

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
"""
For LLMs, we need to generate a model signature: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
Model signatures show the expected input and output types for a model. Which makes quality assurance for downstream serving easier
"""
#use our original text as an example input
data = text_to_summarize
#generate a summary for the output example
output = generate_signature_output(pipe, data)
#infer the signature based on model inputs and outputs
signature = infer_signature(data, output)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an Experiment and Log our Model
# MAGIC Great! Now that we have our model signature, we want create an experiment and log our model. Typically, we log models after finetuning or packaging with other artifacts (more on that later). But for now, we're just going to do a simple `log_model()`run to explore MLflow's functionality 
# MAGIC
# MAGIC Note that, if not set, MLflow automatically sets the `experiment_name` for all MLflow experiments. However, as best practice, you should always name your experiments and model artifacts for better tracking.

# COMMAND ----------

#Create a new mlflow experiment or get the existing one if already exists.
experiment_name = f"/Users/{current_user}/genai-intro-workshop"
mlflow.set_experiment(experiment_name)

#set the name of our model
model_name = "jpeg-mafia"

#get experiment id to pass to the run
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
with mlflow.start_run(experiment_id=experiment_id):
  mlflow.transformers.log_model(pipe, model_name, signature=signature, input_example=data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register our Model to Unity Catalog
# MAGIC Now that we've logged our model, we can register it to Unity Catalog. Typically, we would probably do some additional model comparisons or testing before registering, but we will do so here to demonstrate some functionality

# COMMAND ----------

import mlflow

#grab our most recent run (which logged the model) using our experiment ID
runs = mlflow.search_runs([experiment_id])
last_run_id = runs.sort_values('start_time', ascending=False).iloc[0].run_id

#grab the model URI that's generated from the run
model_uri = f"runs:/{last_run_id}/{model_name}"

#log the model to catalog.schema.model. The schema name referenced below is generated for you in the init script
catalog = dbutils.widgets.get("catalog_name")
schema = schema_name
model_name = "jpeg_mafia"

#set our registry location to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=f"{catalog}.{schema}.{model_name}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View Our Model in Unity Catalog
# MAGIC
# MAGIC Now we can navigate to our `Catalog` and view our model
