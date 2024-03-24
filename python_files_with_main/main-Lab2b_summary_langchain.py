import json
import os
import sys
from datetime import datetime

import boto3
import botocore
from botocore.config import Config
import flax

from io import StringIO
import sys
import textwrap

from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


textGenerationConfigParams = {
    "maxTokenCount":4096,
    "stopSequences":[],
    "temperature":0,
    "topP":1
}
modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider
accept = "application/json"
contentType = "application/json"

shareholder_letter = "./2022-letter-b.txt"


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


class LargeLangModel:
    def __init__(self, boto3_bedrock):
        self._llm = Bedrock(
                model_id=modelId,
                model_kwargs=textGenerationConfigParams,
                client=boto3_bedrock,
            )
	
    def get_llm(self):
        return(self._llm)

def get_file_details(title, file):
    sentance_count = 0
    word_count = 0
    count_char = 0
    sentance_list = file.split('.')
    sentance_count = len(sentance_list)
    word_list = file.split()
    word_count = len(word_list)
    for line in file:
        count_char += len(line)    # note every line is a char. Use 'for line in word_list' to drop count of spaces
    print('The', title, 'file has', sentance_count, 'sentances, ', word_count, 'words and', count_char, 'characters (including spaces)')
    return 


def split_document(llm, letter):
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
    return docs


def summarize(llm, docs):
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
    return output

def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    llm = LargeLangModel(boto3_bedrock).get_llm()
    print(datetime.now())
    with open(shareholder_letter, "r", encoding="utf8") as file:
        letter = file.read()
    docs = split_document(llm, letter)
    output = summarize(llm, docs)
    print('\n')
    get_file_details('original', letter)
    get_file_details('summarized', output)
main()

