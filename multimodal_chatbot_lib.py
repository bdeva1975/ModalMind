import os
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
import base64

# Load environment variables from .env file
load_dotenv()

MAX_MESSAGES = 20

class ChatMessage:
    def __init__(self, role, message_type, text, bytesio=None, image_bytes=None):
        self.role = role
        self.message_type = message_type
        self.text = text
        self.bytesio = bytesio
        self.image_bytes = image_bytes

def get_bytesio_from_bytes(image_bytes):
    return BytesIO(image_bytes)

def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        return image_file.read()

def convert_chat_messages_to_openai_api(chat_messages):
    messages = []
    for chat_msg in chat_messages:
        if chat_msg.message_type == 'image':
            base64_image = base64.b64encode(chat_msg.image_bytes).decode('utf-8')
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({
                "role": chat_msg.role,
                "content": chat_msg.text
            })
    return messages

def chat_with_model(message_history, new_text=None, new_image_bytes=None):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if new_text:
        new_text_message = ChatMessage('user', 'text', text=new_text)
        message_history.append(new_text_message)
    elif new_image_bytes:
        image_bytesio = get_bytesio_from_bytes(new_image_bytes)
        new_image_message = ChatMessage('user', 'image', text=None, bytesio=image_bytesio, image_bytes=new_image_bytes)
        message_history.append(new_image_message)

    if len(message_history) > MAX_MESSAGES:
        message_history = message_history[-MAX_MESSAGES:]

    messages = convert_chat_messages_to_openai_api(message_history)

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use GPT-4 with vision capabilities
            messages=messages,
            max_tokens=2000,
            temperature=0,
            top_p=0.9
        )

        output = chat_completion.choices[0].message.content

        response_message = ChatMessage('assistant', 'text', output)
        message_history.append(response_message)

        return output
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    message_history = []
    
    # Text message example
    response = chat_with_model(message_history, new_text="Hello, can you describe what you see in the next image?")
    print("Bot:", response)

    # Image message example
    # Uncomment and modify the path to use an actual image
    # image_path = "path/to/your/image.jpg"
    # image_bytes = get_bytes_from_file(image_path)
    # response = chat_with_model(message_history, new_image_bytes=image_bytes)
    # print("Bot:", response)

    # Interactive chat loop
    while True:
        user_input = input("You (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat_with_model(message_history, new_text=user_input)
        print("Bot:", response)