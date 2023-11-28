# Databricks notebook source
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
catalog_name = f'genai_workshop_{current_user.split("@")[0].split(".")[0]}'
schema_name = "chatbot"

base_dir = f"/tmp/genai_workshop/{current_user}"
raw_data_dir = f'{base_dir}/raw'
hugging_face_cache = "/tmp/hfcache"

#initialization stuff
print(f"using base_dir {base_dir}")
print(f"Using catalog_name {catalog_name}")
print(f"Using schema_name {schema_name}")
print(f"Using raw_data_dir {raw_data_dir}")
print(f"Using hugging_face_cache {hugging_face_cache}")
