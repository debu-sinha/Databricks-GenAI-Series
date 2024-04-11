# Databricks notebook source
import re

def get_current_user_first_name(email):
    return f"{re.split('[.@]', email)[0]}"

# COMMAND ----------

current_user = spark.sql("SELECT current_user() as username").collect()[0].username
table_suffix = get_current_user_first_name(current_user)
# change catalog_name and schema name to the catalog and schema the students of the workshop can create tables in.
catalog_name = "<replace with your catalog name>"
schema_name = "<replace with your schema name>"

#all the students will have their own tmp dir to work with
base_dir = f"/tmp/genai_workshop/{current_user}"
raw_data_dir = f'{base_dir}/raw'
hugging_face_cache = "/tmp/hfcache"

#initialization stuff
print(f"Using table_suffix {table_suffix}")
print(f"Using hugging_face_cache {hugging_face_cache}")
print(f"using base_dir {base_dir}")
print(f"Using catalog_name {catalog_name}")
print(f"Using schema_name {schema_name}")
print(f"Using raw_data_dir {raw_data_dir}")
print(f"Using hugging_face_cache {hugging_face_cache}")

# COMMAND ----------


