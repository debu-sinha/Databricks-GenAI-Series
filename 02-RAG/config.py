# Databricks notebook source
current_user_email = spark.sql("SELECT current_user() as username").collect()[0].username
current_user=current_user_email.split("@")[0].replace(".","_")

#if you have a pre-determined catalog where the users will be able to create tables and schema change this value to that
catalog_name="genai_workshop"
# catalog_name = f'genai_workshop_{current_user.split("@")[0].split(".")[0]}'

schema_name = f"chatbot_{current_user}"
base_dir = f"/tmp/genai_workshop/{current_user}"
raw_data_dir = f'{base_dir}/raw'
hugging_face_cache = "/tmp/hfcache"
mlflow_experiment_name=f"genai_workshop_{current_user}"
serving_endpoint_name=f"chatbot_{current_user}"

#initialization stuff
print(f"using base_dir {base_dir}")
print(f"Using catalog_name {catalog_name}")
print(f"Using schema_name {schema_name}")
print(f"Using raw_data_dir {raw_data_dir}")
print(f"Using hugging_face_cache {hugging_face_cache}")
