# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', UserWarning)

# COMMAND ----------

from enum import Enum
import subprocess

# Define an Enum for dataset types
class DatasetType(Enum):
    GARDENING = 1
    DATABRICKS = 2

def setup_datasets(dataset_type=DatasetType.DATABRICKS, reset=False, max_documents=None):
    """
    Initializes and configures datasets based on the specified dataset type. This function is capable of 
    handling different types of datasets including Databricks and Gardening Q/A. It supports resetting 
    data directories, catalogs, and schemas, and downloads new data as per the requirements.

    The function operates in several steps:
    1. If 'reset' is True, it resets the directory structure, creates a new catalog and schema.
    2. Depending on the 'dataset_type', it downloads the respective dataset.
    3. For the Databricks dataset, it can limit the number of documents downloaded based on 'max_documents'.

    Parameters:
    - dataset_type (DatasetType, optional): The type of dataset to set up. Default is DatasetType.DATABRICKS.
    - reset (bool, optional): Flag to determine whether to reset the data directories, catalogs, and schemas. Default is False.
    - max_documents (int, optional): The maximum number of documents to download for the Databricks dataset. 
      If None, all documents will be downloaded. Default is None.

    Raises:
    - Exception: If there are any errors during the reset operations or data download process.

    Note:
    - The function internally calls 'download_gardening_dataset' or 'download_databricks_dataset' based on the dataset type.
    - It requires 'dbutils', 'spark', and other dependencies to be pre-configured in the environment.

    Returns:
    - None: This function does not return any value.
    """

    if reset:
        # Reset directory structure and create catalog and schema
        try:
            dbutils.fs.rm(raw_data_dir, True)
            dbutils.fs.mkdirs(raw_data_dir)
            spark.sql(f"USE CATALOG {catalog_name}")
            spark.sql(f"USE SCHEMA {schema_name}")
        except Exception as e:
            print(f"Error during reset operations: {e}")

    # Check the dataset type and download accordingly
    if dataset_type == DatasetType.GARDENING:
        download_gardening_dataset(max_documents)
    elif dataset_type == DatasetType.DATABRICKS:
        download_databricks_dataset(max_documents)

def download_gardening_dataset(max_documents=None):
    from lxml import etree
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    import pandas as pd
    import os
    print(f"Downloading raw Gardening Q/A dataset to /dbfs{raw_data_dir}/gardening")
     # Download and process new data
    try:
        #settting up gardening QA raw dataset

        # the following code is to prevent warning from apt-get
        # Define the repository entry
        REPO_ENTRY="deb [signed-by=/usr/share/keyrings/azul.gpg] https://repos.azul.com/zulu/deb stable main"
        # Define the file where the repository entry will be added
        REPO_FILE="/etc/apt/sources.list.d/zulu.list"
        result=subprocess.run([
            'bash', '-c', 
            f"""
            set -e
            echo "Resetting and downloading new data..."
            apt install gnupg ca-certificates curl
            # Check if the repository entry already exists
            if ! fgrep -q "{REPO_ENTRY}" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
                # Add the repository key
                curl -s https://repos.azul.com/azul-repo.key | sudo gpg --dearmor -o /usr/share/keyrings/azul.gpg
                # Add the repository entry
                echo "{REPO_ENTRY}" | sudo tee "{REPO_FILE}"
            else
                echo "Repository entry already exists."
            fi

            sudo apt-get update
            add-apt-repository universe
            apt-get install -y apt-utils p7zip-full p7zip-rar

            rm -rf /tmp/gardening || true
            mkdir -p /tmp/gardening
            cd /tmp/gardening
            curl -L https://archive.org/download/stackexchange/gardening.stackexchange.com.7z -o gardening.7z
            7z x gardening.7z
            mkdir -p /dbfs{raw_data_dir}/gardening
            cp -f Posts.xml /dbfs{raw_data_dir}/gardening
            """
        ], capture_output=True, text=True)
        # Print the output and any error if occurred
        print("Subprocess Output:\n", result.stdout)
        if result.stderr:
            print("Subprocess Error:\n", result.stderr)
    
        def parse_xml_to_dataframe(xml_file_path):
            """
            Parses the XML file and converts it into a Pandas DataFrame.

            Args:
            - xml_file_path (str): Path to the XML file.

            Returns:
            - DataFrame: Pandas DataFrame containing the parsed XML data.
            """
            # Parse the XML file
            tree = etree.parse(xml_file_path)
            root = tree.getroot()

            # Extract relevant data
            data = []
            for elem in root:
                data.append(elem.attrib)

            # Convert to DataFrame
            return pd.DataFrame(data)
        
        @pandas_udf("string")
        def process_gardening_data_udf(xml_content: pd.Series) -> pd.Series:
            # Process each XML content
            return xml_content.apply(lambda x: parse_xml_to_dataframe(x))
        
        # Assume raw_data_dir is a predefined directory path
        raw_xml_file_path = os.path.join(f"/dbfs{raw_data_dir}/gardening", "Posts.xml")

        # Convert XML to DataFrame
        gardening_df = parse_xml_to_dataframe(raw_xml_file_path)

        # Convert to Spark DataFrame
        spark_gardening_df = spark.createDataFrame(gardening_df)

        # Write to Delta table
        view_name = f"gardening_dataset_raw_{table_suffix}"
        spark_gardening_df.createOrReplaceTempView(view_name)

        print(f"Raw data is now available in view {view_name}")
        
    except Exception as e:
        print(f"Error during data download: {e}")

def download_databricks_dataset(max_documents=None):
    print(f"Downloading raw Databricks dataset.")
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import StringType, StructType, StructField
    import requests
    from bs4 import BeautifulSoup
    import xml.etree.ElementTree as ET
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"

    # Fetch the XML content from sitemap
    response = requests.get(DATABRICKS_SITEMAP_URL)
    root = ET.fromstring(response.content)

    # Find all 'loc' elements (URLs) in the XML
    urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if max_documents:
        urls = urls[:max_documents]

    # Create DataFrame from URLs
    df_urls = spark.createDataFrame(urls, StringType()).toDF("url")

    # Pandas UDF to fetch HTML content for a batch of URLs
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        def fetch_html(url):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_html, urls))
        return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
    @pandas_udf("string")
    def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
        def extract_text(html_content):
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                article_div = soup.find("div", itemprop="articleBody")
                if article_div:
                    return str(article_div).strip()
            return None

        return html_contents.apply(extract_text)

    # Apply UDFs to DataFrame
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))

    # Select and filter non-null results
    view_name = f"databricks_documentation_raw_{table_suffix}"
    final_df = final_df.select("url", "text").filter("text IS NOT NULL")

    final_df.createOrReplaceTempView(view_name)

    print(f"Raw data is now available in view {view_name}")

# COMMAND ----------

import os
import subprocess

def is_folder_empty(path, local=False):
    """
    Check if a specified directory is empty or does not exist.

    This function determines whether a given directory is empty or non-existent. It can
    work with both local file paths and paths on Databricks File System (DBFS) based on the
    'local' flag. The function modifies the path to include a '/dbfs' prefix if the 'local'
    flag is set to False.

    Parameters:
    path (str): The path to the directory to check.
    local (bool, optional): Flag to indicate if the path is on the local file system or DBFS. 
                            Defaults to False (i.e., assumes path is on DBFS).

    Returns:
    bool: True if the directory is empty or does not exist, False otherwise.
    """
    if not local:
        path=f"/dbfs{path}"
        
    if not os.path.isdir(path) or not os.listdir(path):
        print(f"{path} either doesn't exist or is empty.")
        return True
    else:
        return False


# COMMAND ----------

import torch

def reset_gpu():
    """
    Resets the GPU memory if CUDA is available.

    This function checks if CUDA is available in the environment. If so, it performs a series
    of operations to reset the GPU memory. This includes resetting peak memory statistics,
    emptying the cache, and resetting maximum and accumulated memory statistics for each GPU 
    device available in the system. This can be particularly useful in environments where
    long-running processes or multiple consecutive operations might lead to fragmented or 
    inefficiently utilized GPU memory.

    If CUDA is not available, the function simply prints a message stating that no GPU reset
    is required, making the function safe to call in environments without GPU support.
    """

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Resetting the current CUDA device
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Loop through all available GPUs and reset them
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)
            torch.cuda.reset_accumulated_memory_stats(i)
        
        print("GPU memory cleared and reset.")
    else:
        print("CUDA is not available. No GPU reset required.")

# COMMAND ----------

from tensorflow.python.client import device_lib

def get_available_gpus():
    """
    Retrieves the number and names of available GPUs using TensorFlow's device library.

    This function utilizes TensorFlow's device_lib module to list all local devices. It then 
    filters these devices to identify those that are GPUs. The function returns both the 
    count of available GPU devices and their names.

    Returns:
    tuple:
        - int: The number of available GPUs.
        - list of str: The names of each available GPU.

    Note:
    This function is TensorFlow-specific and relies on TensorFlow's ability to interact with
    the underlying hardware. It will only recognize GPUs that TensorFlow is able to use.
    """
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_names), gpu_names

# COMMAND ----------

def reset_vector_db(persist_directory="/tmp/chroma", local=True):
    """
    Resets the vector database by clearing the specified persistence directory.

    This function checks if the specified directory is empty and, based on the result, 
    decides whether to reset the vector database. Resetting involves removing all files 
    in the persistence directory. This operation is resource-intensive and should be used 
    with caution, especially since it currently relies on local storage due to issues with 
    dbfs interfacing in langchain and chromadb APIs.

    Parameters:
    - persist_directory (str): The path to the directory where vector database files are persisted.
                                Default is "/tmp/chroma".
    - local (bool): Flag to indicate if the directory check should be done locally. Default is True.

    Exceptions:
    - Exception: Raises an exception if there is an error in checking the directory's state or 
                 during the reset operation.

    Returns:
    - None
    """

    reset_vdb = True
    try:
        # Assuming is_folder_empty is a predefined function
        reset_vdb = is_folder_empty(persist_directory, local=local)
    except Exception as e:
        print(f"Error checking reset_data widget: {e}")
        reset_vdb = True

    if reset_vdb:
        print("""
            Resetting vector_db

            Please note: This is a resource-intensive operation. Currently, both the langchain and chromadb APIs are encountering issues when interfacing with dbfs, leading us to utilize local storage instead.
            """)

        try:
            import subprocess
            import os

            # Running a simple shell command
            if os.path.exists(persist_directory):
                result = subprocess.run(['rm', '-rf', persist_directory], capture_output=True, text=True)
                print(result.stdout)

        except Exception as e:
            print(f"Error during resetting of Vector DB: {e}")


# COMMAND ----------

from transformers import pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain



def build_qa_chain(model_name="databricks/dolly-v2-7b", 
                   torch_dtype=torch.bfloat16, 
                   max_new_tokens=256, 
                   top_k=50,
                   prompt_template=None, 
                   chain_type="stuff",
                   verbose=False,
                   **pipeline_kwargs):
    """
    Builds and returns a QA chain using the specified model, parameters, and optional prompt template.

    Args:
        model_name (str): Name of the HuggingFace model to use.
        torch_dtype (torch.dtype): Data type for model tensors.
        max_new_tokens (int): Maximum new tokens to generate.
        top_k (int): Top-k sampling's k value.
        prompt_template (PromptTemplate, optional): Custom prompt template. If None, uses a default template.
        chain_type (str): Type of the QA chain.
        verbose (bool): Verbose output flag.
        **pipeline_kwargs: Additional keyword arguments for the HuggingFace pipeline.

    Returns:
        A QA chain object.
    """
 
    # Create the pipeline with specified parameters
    instruct_pipeline = pipeline(
        model=model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        **pipeline_kwargs
    )

    # Define a default prompt template if none provided
    if not prompt_template:
        template_text = """..."""  # Default template text
        prompt_template = PromptTemplate(input_variables=['context', 'question'], template=template_text)

    # Create a HuggingFace pipeline
    hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

    # Return the QA chain
    return load_qa_chain(llm=hf_pipe, chain_type=chain_type, prompt=prompt_template, verbose=verbose)


# COMMAND ----------

def format_and_display_chat_response(question, result):
    """
    Generates an HTML-formatted response for a given question using a pre-computed result.

    This function takes a question and a result object, formats them into a chat-style 
    HTML representation, and displays it. The result object is expected to contain the 
    answer and the input documents used to generate the answer. The function formats 
    the question and answer as chat bubbles and lists the input documents as references.

    Parameters:
    - question (str): The question to be answered.
    - result (dict): The result object containing 'output_text' and 'input_documents'.
                     'output_text' is the answer to the question, and 'input_documents' 
                     is a list of documents that were used to generate the answer.

    Returns:
    - None: The function outputs the formatted HTML directly using displayHTML.
    """

     # Starting the HTML formatting for the chat dialog
    result_html = "<div style='font-family: Arial, sans-serif; max-width: 600px; margin: auto;'>"
    
    # Formatting the question as a chat bubble
    result_html += f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 15px; margin: 10px 0;'>"
    result_html += f"<p style='margin: 0;'><strong>Question:</strong> {question}</p>"
    result_html += "</div>"

    # Formatting the answer as a chat bubble
    result_html += f"<div style='background-color: #e0e0ff; padding: 10px; border-radius: 15px; margin: 10px 0;'>"
    result_html += f"<p style='margin: 0;'><strong>Answer:</strong> {result['output_text']}</p>"
    result_html += "</div>"

    # Adding a divider
    result_html += "<hr/>"

    # Formatting the input documents as references
    for d in result["input_documents"]:
        source = d.metadata["url"]
        result_html += f"<div style='background-color: #f8f8f8; padding: 10px; border-radius: 15px; margin: 10px 0;'>"
        result_html += f"<p style='margin: 0;'>{d.page_content}<br/>"
        result_html += f"(Source: <a href='{source}'>{source}</a>)</p>"
        result_html += "</div>"

    # Closing the HTML formatting
    result_html += "</div>"

    displayHTML(result_html)
