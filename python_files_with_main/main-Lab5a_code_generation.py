import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
from io import StringIO
import textwrap

import boto3
import botocore
from botocore.config import Config

#modelId = "amazon.titan-tg1-large"  
modelId = "anthropic.claude-v2"  # claude gives more accurate results than titan
accept = "application/json"
contentType = "application/json"

	
prompt_data_python = """

Human: You have a CSV, sales.csv, with columns:
- date (YYYY-MM-DD)
- product_id
- price
- units_sold

Create a python program to analyze the sales data from a CSV file. The program should be able to read the data, and determine the below. Remember to include python code to print the determined values to the terminal. Cast the 'price' values and 'units_sold values to int. Cast the 'date' and 'product_id' as str.

- Total revenue for the year
- The product_id with the highest revenue
- The date with the highest revenue

Ensure the python code is syntactically correct, bug-free, optimized, not span multiple lines unnessarily, and prefer to use standard libraries. Return only python code without any surrounding text, explanation or context.

Assistant:
"""

prompt_data_pyspark = """

Human: You have a CSV, sales.csv, with columns:
- date (YYYY-MM-DD)
- product_id
- price
- units_sold

Create a pyspark program to analyze the sales data from a CSV file. The program should be able to read the data, and determine the below. Remember to include python code to print the determined values to the terminal. Cast the 'price' values and 'units_sold values to int. Cast the 'date' and 'product_id' as str.

- Total revenue for the year
- The product_id with the highest revenue
- The date with the highest revenue

Ensure the pyspark code is syntactically correct, bug-free, optimized, not span multiple lines unnessarily, and prefer to use standard libraries. Return only python code without any surrounding text, explanation or context.

Assistant:
"""

prompt_data_sql = """

Human: AnyCompany has a database with a table named sales_data containing sales records. The table has following columns:
- date (YYYY-MM-DD)
- product_id
- price
- units_sold

Can you generate SQL queries for the below: 
- Identify the top 5 best selling products by total sales for the year 2023
- Calculate the monthly average sales for the year 2023

Assistant:
"""

def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

def get_bedrock_client():
    service_name='bedrock-runtime'
    target_region = "us-west-2"
    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}
    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)
    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )
    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


def create_code(prompt_data, boto3_bedrock, modelId, accept, contentType):
    # Claude - Body Syntax
    body = json.dumps({
                    "prompt": prompt_data,
                    "max_tokens_to_sample":4096,
                    "temperature":0.5,
                    "top_k":250,
                    "top_p":0.5,
                    "stop_sequences": ["\n\nHuman:"]
                  }) 
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    #print_ww(response_body.get('results')[0]['outputText'])     	# amazon.titan-tg1-large
    print_ww(response_body.get('completion'))						# anthropic.claude-v2
    return
	
	
def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    prompt_data = prompt_data_python
    create_code(prompt_data, boto3_bedrock, modelId, accept, contentType)


main()
