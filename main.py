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
from langchain.schema import HumanMessage, AIMessage, SystemMessage

print(sys.path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials and config for directory and prompt settings
load_credentials()
config = load_config()

async def chat(message: str, history: List[tuple[str, str]], model_choice: str, history_flag: bool, stream: bool = False):
    logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {model_choice}")
    model = get_model(model_choice)
    logger.info(f"Model instantiated: {model}")
    if stream:
        result = [chunk async for chunk in model.chat(message, history if history_flag else [], stream=True)]
    else:
        result = [chunk async for chunk in model.chat(message, history if history_flag else [], stream=False)]
    return result

async def prompt(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, stream: bool = False):
    logger.info(f"Prompt function called with message: {message}, model_choice: {model_choice}, prompt_info: {prompt_info}")
    model = get_model(model_choice)
    system_prompt = get_prompt(prompt_info)
    logger.info(f"Model instantiated: {model}, system_prompt: {system_prompt}")
    if stream:
        result = [chunk async for chunk in model.prompt(message, system_prompt, stream=True)]
    else:
        result = [chunk async for chunk in model.prompt(message, system_prompt, stream=False)]
    return result

async def process_image(image: bytes, question: str, model_choice: str, stream: bool = False):
    logger.info(f"Process image called with question: {question}, model_choice: {model_choice}")
    if image is None:
        return "Please upload an image first."
    
    model = get_model(model_choice)
    logger.info(f"Model instantiated: {model}")
    if stream:
        result = [chunk async for chunk in model.image_chat(image, question, stream=True)]
    else:
        result = [chunk async for chunk in model.image_chat(image, question, stream=False)]
    return result

# Wrapping async functions for Gradio
def chat_wrapper(*args, **kwargs):
    async def run():
        result = await chat(*args, **kwargs, stream=True)
        return "".join(result)
    return asyncio.run(run())

def prompt_wrapper(*args, **kwargs):
    async def run():
        result = await prompt(*args, **kwargs, stream=True)
        return "".join(result)
    return asyncio.run(run())

def process_image_wrapper(*args, **kwargs):
    async def run():
        result = await process_image(*args, **kwargs, stream=True)
        return "".join(result)
    return asyncio.run(run())
    
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
        process_image_wrapper,
        inputs=[image_input, question_input, model_choice],
        outputs=[output]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch()