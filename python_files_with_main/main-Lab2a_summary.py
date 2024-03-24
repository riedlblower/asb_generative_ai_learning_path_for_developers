import json
import os
import sys

import boto3
import botocore
from botocore.config import Config

from io import StringIO
import sys
import textwrap


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

def set_parameters():
    textGenerationConfigParams = {
        "maxTokenCount":4096,
        "stopSequences":[],
        "temperature":0,
        "topP":1
    }
    modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"
    return textGenerationConfigParams, modelId, accept, contentType



def set_prompt():
    # create the prompt
    
    prompt = """
    Please provide a summary of the following text. Do not add any information that is not mentioned in the text below. 
    <text>
    AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
    a new service that makes FMs from AI21 Labs, Anthropic, Stability AI, and Amazon accessible via an API. \
    Bedrock is the easiest way for customers to build and scale generative AI-based applications using FMs, \
    democratizing access for all builders. Bedrock will offer the ability to access a range of powerful FMs \
    for text and images—including Amazons Titan FMs, which consist of two new LLMs we’re also announcing \
    today—through a scalable, reliable, and secure AWS managed service. With Bedrock’s serverless experience, \
    customers can easily find the right model for what they’re trying to get done, get started quickly, privately \
    customize FMs with their own data, and easily integrate and deploy them into their applications using the AWS \
    tools and capabilities they are familiar with, without having to manage any infrastructure (including integrations \
    with Amazon SageMaker ML features like Experiments to test different models and Pipelines to manage their FMs at scale).
    </text>    
    """
    return prompt

def set_body(prompt_data, textGenerationConfigParams):
    body = json.dumps({
        "inputText": prompt_data, 
        "textGenerationConfig": textGenerationConfigParams
        })
    return body


def generate_summary(boto3_bedrock, body, modelId, accept, contentType):
    try:
        response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        print_ww(response_body.get('results')[0].get('outputText'))
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
               print(f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                     \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                     \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        else:
            raise error
    return
		
		
def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    parameters, modelId, accept, contentType = set_parameters()
    prompt_data = set_prompt()
    body = set_body(prompt_data, parameters)
    generate_summary(boto3_bedrock, body, modelId, accept, contentType)

main()