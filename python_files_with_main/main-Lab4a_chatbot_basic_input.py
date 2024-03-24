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

#modelId = "amazon.titan-tg1-large"
modelId = "anthropic.claude-v2"
#titan_llm = Bedrock(model_id=modelId, client=boto3_bedrock)
#titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}

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

def get_titan_llm(modelId, boto3_bedrock):
    titan_llm = Bedrock(model_id=modelId, client=boto3_bedrock)
#    titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}
    titan_llm.model_kwargs = {'max_tokens_to_sample': 1000}
    return titan_llm

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

#def organize_memory():
#    memory = ConversationBufferMemory()
#    memory.human_prefix = "User"
#    memory.ai_prefix = "Bot"
#    return memory

def organize_conversation(titan_llm):
#    conversation = ConversationChain(llm=titan_llm, verbose=True, memory=memory)
    conversation = ConversationChain(llm=titan_llm, verbose=True)
    conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. The assistant is talkative and provides lots of specific details from it's context.\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""
    return conversation

def have_conversation(conversation):
    try:
        print_ww(conversation.predict(input="Hi there!"))
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

    for i in range(1, 100):
        question = input("What is your question? Type Q to quit ")
        if (question == 'quit') or (question == 'q') or (question == 'Q'):
            break
        print_ww(conversation.predict(input=question))
        i = i + 1
    print_ww(conversation.predict(input="That's all, thank you!"))
    return


def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    titan_llm = get_titan_llm(modelId, boto3_bedrock)
#    memory = organize_memory()
    conversation = organize_conversation(titan_llm)
    have_conversation(conversation)

main()