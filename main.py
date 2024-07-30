import logging
import os
import json
from io import BytesIO
import base64
import sys
from PIL import Image
import gradio as gr
from ai_model_interface import get_model, load_credentials, load_config, get_prompt, get_prompt_list
import asyncio
from typing import List, Union, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage

print(sys.path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials and config for directory and prompt settings
load_credentials()
config = load_config()

async def chat(message: str, history: List[tuple[str, str]], model_choice: str, history_flag: bool):
    model = get_model(model_choice)
    return await model.chat(message, history if history_flag else [], stream=True)

async def prompt(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str):
    model = get_model(model_choice)
    system_prompt = get_prompt(prompt_info)
    return await model.prompt(message, system_prompt, stream=True)

async def process_image(image: bytes, question: str, model_choice: str):
    if image is None:
        return "Please upload an image first."
    
    model = get_model(model_choice)
    return await model.image_chat(image, question)

# If you're using Gradio, you might need to wrap these async functions
def chat_wrapper(*args, **kwargs):
    return asyncio.run(chat(*args, **kwargs))

def prompt_wrapper(*args, **kwargs):
    return asyncio.run(prompt(*args, **kwargs))

def process_image_wrapper(*args, **kwargs):
    return asyncio.run(process_image(*args, **kwargs))

    
with gr.Blocks() as demo:
    gr.Markdown("# Image Question Answering")
    gr.Markdown("Upload an image and ask questions about it using your choice of model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB")
            model_choice = gr.Dropdown(
                ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                label="Choose Model",
                value="Ollama (LLaVA)"
            )
            question_input = gr.Textbox(label="Ask a question about the image")
            submit_btn = gr.Button("Submit")
        
        with gr.Column(scale=1):
            output = gr.Textbox(label="Response", lines=10)
            
    # Process image when submit button is clicked
    submit_btn.click(
        process_image,
        inputs=[image_input, question_input, model_choice],
        outputs=[output]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch()