import logging
import os
import json
from io import BytesIO
import base64
import sys
from PIL import Image
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

print(sys.path)

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

# Call the function to load credentials
load_credentials()

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_model(choice):
    if choice == "Ollama (LLaVA)":
        return ChatOllama(base_url="http://localhost:11434", model="llava") 
    elif choice == "OpenAI GPT-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", max_tokens=300)
    else:
        raise ValueError(f"Invalid model choice: {choice}")

def process_image(image, question, model_choice):
    if image is None:
        logger.warning("No image uploaded")
        return "Please upload an image first."

    logger.info(f"Processing image with model: {model_choice}")
    image_b64 = convert_to_base64(image)

    try:
        model = get_model(model_choice)
        
        if model_choice == "OpenAI GPT-4o-mini":
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                )
            ]
        else:  # Ollama (LLaVA)
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                    ]
                )
            ]

        response = model.invoke(messages)
        logger.info("Successfully processed image and generated response")
        return response.content

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return f"An error occurred: {str(e)}"
    
with gr.Blocks() as demo:
    gr.Markdown("# Image Question Answering")
    gr.Markdown("Upload an image and ask questions about it using your choice of model.")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB")
        model_choice = gr.Radio(
            ["Ollama (LLaVA)", "OpenAI GPT-4o-mini"],
            label="Choose Model",
            value="Ollama (LLaVA)"
        )
    
    question_input = gr.Textbox(label="Ask a question about the image")
    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Response")

    # Process image when submit button is clicked
    submit_btn.click(
        process_image,
        inputs=[image_input, question_input, model_choice],
        outputs=[output]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch()