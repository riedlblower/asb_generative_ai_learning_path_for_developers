import tkinter as tk
import random

responses = [
    "Hello! How can I help you today?",
    "What's on your mind?",
    "I'm here to assist you. What do you need?",
    "How can I assist you?",
    "What can I help you with?",
]

def chatbot_response(user_input):
  # Normalize the user's input
  user_input = user_input.lower()

  # Check for specific keywords in the user's input
  if "movie" in user_input:
    return "I recommend checking out the IMDb website for movie recommendations. They have a wide variety of genres and ratings to choose from."
  elif "weather" in user_input:
    return "You can check the weather by using a weather website or app. Some popular ones include Weather.com and The Weather Channel app."
  elif "news" in user_input:
    return "There are many websites and apps that offer the latest news updates, such as CNN, Fox News, and NBC News."
  elif "joke" in user_input:
    return "Why couldn't the bicycle stand up by itself? Because it was two-tired!"
  else:
    # If no keywords are detected, select a random response from the list
    return random.choice(responses)



def send_message(input_field, text_area):
  # Get the user's input
  user_input = input_field.get()

  # Clear the input field
  input_field.delete(0, tk.END)

  # Generate a response from the chatbot
  response = chatbot_response(user_input)

  # Display the response in the chatbot's text area
  text_area.insert(tk.END, f"User: {user_input}\n")
  text_area.insert(tk.END, f"Chatbot: {response}\n")


def create_textbox():
    root = tk.Tk()
    root.title("Chatbot")

    # Create the chatbot's text area
    text_area = tk.Text(root, bg="white", width=50, height=20)
    text_area.pack()
    
    # Create the user's input field
    input_field = tk.Entry(root, width=50)
    input_field.pack()
    return root, text_area, input_field
    

	
def create_send_button_and_have_conversation(root, text_area, input_field):
    send_button = tk.Button(root, text="Send", command=lambda: send_message(input_field, text_area))
    send_button.pack()
    root.mainloop()


def main():
    root, text_area, input_field = create_textbox()
    create_send_button_and_have_conversation(root, text_area, input_field)

main()