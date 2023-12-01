# Databricks notebook source
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "mosaicml/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device_map='auto'
)

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output

data = "Why is my Spark join slow?"
output = generate_signature_output(pipe, data)
signature = infer_signature(data, output)

mlflow.transformers.log_model(pipe, "mpt-7b-doan-demo", signature=signature, input_example=data)

# COMMAND ----------

import mlflow
catalog = "doan_demo_catalog"
schema = "language_models"
model_name = "mpt-7b-doan-demo"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri="dbfs:/databricks/mlflow-tracking/05739e2a092c4d60b3e03bc138b09e07/bd85c2dec62947858caffcabc862ab98/artifacts/mpt-7b-doan-demo",
    name=f"{catalog}.{schema}.{model_name}"
)
