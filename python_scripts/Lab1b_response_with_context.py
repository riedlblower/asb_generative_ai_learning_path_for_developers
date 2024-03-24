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

from langchain.llms.bedrock import Bedrock

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

inference_modifier = {'maxTokenCount':4096, 
                      "temperature":0.5,
                      "topP":1,
                      "stopSequences": []
                     }
#inference_modifier = {'max_tokens_to_sample':4096, 
#                      "temperature":0.5,
#                      "top_k":250,
#                      "top_p":1,
#                      "stop_sequences": ["\n\nHuman"]
#                     }
#anthropic.claude-v2
textgen_llm = Bedrock(model_id = "amazon.titan-tg1-large",
                    client = boto3_bedrock, 
                    model_kwargs = inference_modifier 
                    )

from langchain.prompts import PromptTemplate

# Create a prompt template that has multiple input variables
multi_var_prompt = PromptTemplate(
    input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"], 
    template="""

Human: Create an apology email from the Service Manager {customerServiceManager} to {customerName} in response to the following feedback that was received from the customer: 
<customer_feedback>
{feedbackFromCustomer}
</customer_feedback>

Assistant:"""
)

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

num_tokens = textgen_llm.get_num_tokens(prompt)
print(f"Our prompt has {num_tokens} tokens")


response = textgen_llm(prompt)

email = response[response.index('\n')+1:]

print_ww(email)

