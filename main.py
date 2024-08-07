import logging
import os
import json
import yaml
from io import BytesIO
import base64
import sys
from PIL import Image
import gradio as gr
import asyncio
from typing import List, Union, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from ai_model_interface.config.credentials import get_api_key
from ai_model_interface import get_model, load_credentials, load_config, format_prompt, get_prompt, get_prompt_list, update_prompt_list

print(sys.path)

# Set up logging to only show warnings and errors
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials and config for directory and prompt settings
load_credentials()
config = load_config()

# ai_model_interface/config/settings.py
def load_config() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

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
    
    # Format the prompt using format_prompt function
    formatted_prompt = format_prompt(system_prompt, message, prompt_info)
    
    if stream:
        result = [chunk async for chunk in model.prompt(formatted_prompt, system_prompt, stream=True)]
    else:
        result = [chunk async for chunk in model.prompt(formatted_prompt, system_prompt, stream=False)]
    return result

async def process_image(image: Image.Image, question: str, model_choice: str, stream: bool = False):
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
def chat_wrapper(message, history, model_choice, history_flag):
    async def run():
        result = await chat(message, history, model_choice, history_flag, stream=True)
        return ''.join(result)
    return asyncio.run(run())

def prompt_wrapper(message, history, model_choice, prompt_info):
    async def run():
        result = await prompt(message, history, model_choice, prompt_info, stream=True)
        return ''.join(result)
    return asyncio.run(run())

def process_image_wrapper(message, history, image, model_choice, history_flag):
    if not image:
        return "Please upload an image first."
    
    async def run():
        result = await process_image(image, message, model_choice, stream=True)
        return ''.join(result)
    return asyncio.run(run())

def conversation_wrapper(user_input, model_choice, chat_history_flag):
    # Get the conversation history and formatted history from your model instance
    conversation_history = model_choice.get_conversation_history()
    
    if chat_history_flag.value:
        formatted_history = model_choice._format_history(conversation_history)
        
        copy_to_clipboard(formatted_history)
        
def copy_conversation_js():
    return """
    <script>
      document.getElementById("copy-btn").addEventListener('click', function() {
        conversation_wrapper();
      });
    </script>
    """
           
def clear_chat():
    return None

def clear_vision_chat():
    return None, None, gr.update(value=None)
    
with gr.Blocks() as demo:
    gr.Markdown("# Langchain Working Bench")
    gr.Markdown("### Chat with LLM's of choice and reuse prompts to get work done.")

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    chat_history_flag = gr.Checkbox(label="Include conversation history", value=True)
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Ollama (Deepseek-coder-v2)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                with gr.Column(scale=4):
                    chat_bot = gr.Chatbot(height=600, show_copy_button=True)
                    chat_text_box = gr.Textbox(label="Chat input", placeholder="Type your message here...")
                    gr.ChatInterface(
                        fn=chat_wrapper,
                        chatbot=chat_bot,
                        textbox=chat_text_box,
                        additional_inputs=[model_choice, chat_history_flag],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

        with gr.Tab("Prompting"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Language",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(choices=get_prompt_list("english"), label="Prompt Selection", interactive=True)
                    history_flag_prompt = gr.Checkbox(label="Include conversation history", value=True)

                with gr.Column(scale=4):
                    prompt_chat_bot = gr.Chatbot(height=600, show_copy_button=True)
                    prompt_text_box = gr.Textbox(label="Prompt input", placeholder="Type your prompt here...")
                    gr.ChatInterface(
                        fn=prompt_wrapper,
                        chatbot=prompt_chat_bot,
                        textbox=prompt_text_box,
                        additional_inputs=[model_choice, prompt_info],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

        with gr.Tab("Vision Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB")
                    model_choice = gr.Dropdown(
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLaVA)"
                    )
                    history_flag_vision = gr.Checkbox(label="Include conversation history", value=True)

                with gr.Column(scale=4):
                    vision_chatbot = gr.Chatbot(height=600, show_copy_button=True)
                    vision_question_input = gr.Textbox(label="Ask about the image", placeholder="Type your question about the image here...")
                    gr.ChatInterface(
                        fn=process_image_wrapper,
                        chatbot=vision_chatbot,
                        textbox=vision_question_input,
                        additional_inputs=[image_input, model_choice, history_flag_vision],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear"
                        )

            vision_clear_btn = gr.Button("Clear All")
            vision_clear_btn.click(
                fn=clear_vision_chat,
                inputs=[],
                outputs=[vision_chatbot, vision_question_input, image_input]
            )

    language_choice.change(fn=update_prompt_list, inputs=[language_choice], outputs=[prompt_info])
    copy_btn = gr.Button("Copy Conversation", elem_id="copy-btn")

    gr.HTML(copy_conversation_js())

    copy_btn.click(
        fn=conversation_wrapper,
        inputs=[model_choice],
        outputs=[]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True)