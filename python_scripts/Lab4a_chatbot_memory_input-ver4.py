import boto3
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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


bedrock_client = boto3.client(service_name = 'bedrock-runtime')
model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}
llm = Bedrock(model_id = "anthropic.claude-v2", model_kwargs = model_parameter, client = bedrock_client) 
memory = ConversationBufferMemory(return_messages = True)

conversation = ConversationChain(llm=llm, verbose=False, memory=memory)

claude_prompt = PromptTemplate.from_template("""System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. The assistant is talkative and provides lots of specific details from it's context.

Current conversation:
<conversation_history>
{history}
</conversation_history>

Here is the human's next reply:
<human_reply>
{input}
</human_reply>

Assistant:
""")

conversation.prompt = claude_prompt


for i in range(1, 100):
    question = input("What is your question? Type Q to quit ")
    if (question == 'quit') or (question == 'q') or (question == 'Q'):
        break
    print_ww(conversation.predict(input=question))
    i = i + 1

memory.load_memory_variables({})   # only useful if verbose=True

print_ww(conversation.predict(input="That's all, thank you!"))

