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


boto3_bedrock = get_bedrock_client() # configure_environment()

# create the prompt
prompt_data = """
Command: Write an email from Bob, Customer Service Manager, to the customer "John Doe" 
who provided negative feedback on the service provided by our customer support 
engineer"""


body = json.dumps({
    "inputText": prompt_data, 
    "textGenerationConfig":{
        "maxTokenCount":4096,
        "stopSequences":[],
        "temperature":0,
        "topP":0.9
        }
    })

modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

output = []
try:
    
    response = boto3_bedrock.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
    stream = response.get('body')
    
    i = 1
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['outputText']
                output.append(text)
                print(f'\t\t\x1b[31m**Chunk {i}**\x1b[0m\n{text}\n')
                i+=1
            
except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error
		
		
print('\t\t\x1b[31m**COMPLETE OUTPUT**\x1b[0m\n')
complete_output = ''.join(output)
print(complete_output)