import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
from datetime import datetime
from io import StringIO
import textwrap

import boto3
import botocore
from botocore.config import Config

from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


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
llm = Bedrock(
    model_id=modelId,
    model_kwargs={
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    },
    client=boto3_bedrock,
)
print(datetime.now())
print(llm)
print(datetime.now(), "this test can take a minute or so to execute")

shareholder_letter = "./2022-letter-b.txt"

with open(shareholder_letter, "r", encoding="utf8") as file:
    letter = file.read()
    
llm.get_num_tokens(letter)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])

num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)


print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

# Set verbose=True if you want to see the prompts being used
summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

output = ""
try:
    
    output = summary_chain.run(docs)

except ValueError as error:
    if  "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error

print_ww(output.strip())
print(datetime.now())
