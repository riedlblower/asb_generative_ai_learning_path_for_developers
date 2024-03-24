import json
import os
import sys

import boto3
import botocore
from botocore.config import Config

from io import StringIO
import sys
import textwrap

from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate



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


def get_parameters():
    inference_modifier = {
        "maxTokenCount":4096,
        "stopSequences":[],
        "temperature":0.5,
        "topP":1
    }
    modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider					
    return inference_modifier, modelId


def get_textgen_llm(modelId, boto3_bedrock, inference_modifier):
    textgen_llm = Bedrock(model_id = "amazon.titan-tg1-large",
                        client = boto3_bedrock, 
                        model_kwargs = inference_modifier 
                        )
    return textgen_llm

def create_prompt_template_with_multiple_input_variables():
    # Create a prompt template that has multiple input variables
    multi_var_prompt_template = PromptTemplate(
        input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"], 
        template="""
    
    Human: Create an apology email from the Service Manager {customerServiceManager} to {customerName} in response to the following feedback that was received from the customer: 
    <customer_feedback>
    {feedbackFromCustomer}
    </customer_feedback>
    
    Assistant:"""
    )
    return multi_var_prompt_template

def pass_variables_into_prompt(multi_var_prompt):
    # Pass in values to the input variables
    prompt = multi_var_prompt.format(customerServiceManager="Bob", 
                                    customerName="John Doe", 
                                    feedbackFromCustomer="""Hello Bob,
        I am very disappointed with the recent experience I had when I called your customer support.
        I was expecting an immediate call back but it took three days for us to get a call back.
        The first suggestion to fix the problem was incorrect. Ultimately the problem was fixed after three days.
        We are very unhappy with the response provided and may consider taking our business elsewhere.
        """
        )
    return prompt

def create_response(prompt, textgen_llm):
    num_tokens = textgen_llm.get_num_tokens(prompt)
    print(f"Our prompt has {num_tokens} tokens")
    response = textgen_llm(prompt)
    email = response[response.index('\n')+1:]
    print_ww(email)
    return




def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    inference_modifier, modelId = get_parameters()
    textgen_llm = get_textgen_llm(modelId, boto3_bedrock, inference_modifier)
    multi_var_prompt_template = create_prompt_template_with_multiple_input_variables()
    prompt = pass_variables_into_prompt(multi_var_prompt_template)
    create_response(prompt, textgen_llm)

main()