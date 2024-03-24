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

import tkinter as tk

modelId = "amazon.titan-tg1-large"
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
    titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}
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

def organize_memory():
    memory = ConversationBufferMemory()
    memory.human_prefix = "User"
    memory.ai_prefix = "Bot"
    memory.chat_memory.add_user_message("You will be acting as a career coach. Your goal is to give career advice to users")
    memory.chat_memory.add_ai_message("I am career coach and give career advice")
    return memory

def organize_conversation(titan_llm, memory):
    conversation = ConversationChain(llm=titan_llm, verbose=False, memory=memory)
    ##conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable career coach and a user. The coach is talkative and provides lots of specific details from it's context.\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""
    #conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful career coach and a user. The coach is talkative and provides lots of specific details from it's context.\n{history}\n{input}"""
    return conversation

class introduction:
    def __init__(self):
        self._intro = """The following is a friendly conversation between a career coach and a user. The career coach is talkative and provides lots of specific details from it's context. Please ask a question and press the Send button. Remember sending quit exits the program. Some responses take time so do be patient\n"""

    def get_intro(self):
	    return self._intro

def send_message(input_field, text_area, conversation, root):
  # Get the user's input
  user_input = input_field.get()
  if user_input == 'quit':
    root.destroy()
    return 
	
  # Clear the input field
  input_field.delete(0, tk.END)

  # Generate a response from the chatbot
  response = conversation.predict(input = user_input)

  # Display the response in the chatbot's text area
  text_area.insert(tk.END, f"User: {user_input}\n")
  text_area.insert(tk.END, f"Chatbot: {response}\n")
  return
  
def create_textbox():
    root = tk.Tk()
    root.title("AI Chatbot - Press the Send button to ask a question and (sending quit exits the program)")

    # Create the chatbot's text area
    text_area = tk.Text(root, bg="white", width=100, height=30)
    intro = introduction()
    text_area.insert(tk.END, f"{intro.get_intro()}\n")
    text_area.pack()
    
    # Create the user's input field
    input_field = tk.Entry(root, width=50)
    input_field.pack()
    return root, text_area, input_field
    
def create_send_button_and_have_conversation(root, text_area, input_field, conversation):
    send_button = tk.Button(root, text="Send", command=lambda: send_message(input_field, text_area, conversation, root))
    send_button.pack()
    root.mainloop()


def main():
    boto3_bedrock = get_bedrock_client() # configure_environment()
    titan_llm = get_titan_llm(modelId, boto3_bedrock)
    memory = organize_memory()
    conversation = organize_conversation(titan_llm, memory)
    root, text_area, input_field = create_textbox()
    create_send_button_and_have_conversation(root, text_area, input_field, conversation)

main()