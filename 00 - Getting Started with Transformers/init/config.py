# Databricks notebook source
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging

# COMMAND ----------

dbutils.widgets.text("catalog_name","main")
catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

current_user = spark.sql("SELECT current_user() as username").collect()[0].username
schema_name = f'genai_workshop_{current_user.split("@")[0].split(".")[0]}'

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# COMMAND ----------

xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)

# COMMAND ----------


