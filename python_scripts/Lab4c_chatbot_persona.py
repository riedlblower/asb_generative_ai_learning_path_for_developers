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


from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory


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
modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=boto3_bedrock)
titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}

memory = ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"


memory.chat_memory.add_user_message("You will be acting as a career coach. Your goal is to give career advice to users")
memory.chat_memory.add_ai_message("I am career coach and give career advice")
titan_llm = Bedrock(model_id="amazon.titan-tg1-large",client=boto3_bedrock)
conversation = ConversationChain(
     llm=titan_llm, verbose=True, memory=memory
)

print_ww(conversation.predict(input="What are the career options in AI?"))

conversation.verbose = False
print_ww(conversation.predict(input="How to fix my car?"))