import gradio as gr
from PIL import Image
import base64
from io import BytesIO
import os
import json
import logging
import langchain
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage, SystemMessage, BaseMessage, AIMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_credentials():
    # Try to get the path from an environment variable
    cred_path = os.getenv('CREDENTIALS_PATH')
    
    if not cred_path:
        # Fallback to a relative path
        cred_path = '../../../credentials.json'
    
    try:
        with open(cred_path, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
        os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']
        logger.info("Credentials loaded successfully")
    except FileNotFoundError:
        logger.error(f"Credentials file not found at {cred_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in credentials file at {cred_path}")
        raise
    except KeyError:
        logger.error("'openai_api_key' not found in credentials file")
        raise
    logger.info("Credentials loaded successfully")

# Call the function to load credentials
load_credentials()

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def process_image(image, question):
    if image is None:
        logger.warning("No image uploaded")
        return "Please upload an image first."

    logger.info("Processing image with model: llava")
    image_b64 = convert_to_base64(image)

    try:
        chat_model = ChatOllama(base_url="http://localhost:11434", model="llava")

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            )
        ]

        response = chat_model.invoke(messages)
        logger.info("Successfully processed image and generated response")
        return response.content

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Ask a question about the image")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Image Question Answering",
    description="Upload an image and ask questions about it using Ollama (LLaVA)."
)

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    iface.launch()