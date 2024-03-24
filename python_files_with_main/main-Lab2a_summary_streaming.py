import json
import os
import sys

import boto3
import botocore
from botocore.config import Config

from io import StringIO
import sys
import textwrap

textGenerationConfigParams = {
    "maxTokenCount":4096,
    "stopSequences":[],
    "temperature":0,
    "topP":1
}
modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider
accept = "application/json"
contentType = "application/json"



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

# https://realpython.com/python-getter-setter/#:~:text=Even%20though%20the%20getter%20and,of%20lines%20in%20your%20code.
# https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters
# https://www.geeksforgeeks.org/getter-and-setter-in-python/


class Prompt:
    def __init__(self):
        self._prompt = """
        Please provide a summary of the following text. 
        
        <text>
    `    AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
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
	
    def get_prompt(self):
        return self._prompt

class Body:
    def __init__(self, prompt_data, textGenerationConfigParams):
	    self._body = json.dumps({
        "inputText": prompt_data, 
        "textGenerationConfig": textGenerationConfigParams
        })

    def get_body(self):
        return(self._body)

    def set_body(self, prompt_data, textGenerationConfigParams):
	    self._body = json.dumps({
        "inputText": prompt_data, 
        "textGenerationConfig": textGenerationConfigParams
        })
	    


def generate_streaming_summary(boto3_bedrock, body):
    response = boto3_bedrock.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
    stream = response.get('body')
    	
    output = ''
    i = 1
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['outputText']
                print("chunk", i)
                print(text)
                output = output + text
                #output.append(text)			
                print("\n")			
                i+=1
    
    print("full output")
    print(output)
    return		
		
		
def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    prompt = Prompt()
    body = Body(prompt.get_prompt(), textGenerationConfigParams)
    generate_streaming_summary(boto3_bedrock, body.get_body())

main()