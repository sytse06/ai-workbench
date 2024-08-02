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
from ai_model_interface.config.credentials import get_api_key

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
        return result
    return asyncio.run(run())

def prompt_wrapper(message, history, model_choice, prompt_info):
    async def run():
        result = await prompt(message, history, model_choice, prompt_info, stream=True)
        return result
    return asyncio.run(run())

def process_image_wrapper(image, question, model_choice):
    async def run():
        result = await process_image(image, question, model_choice, stream=True)
        return result
    return asyncio.run(run())

def copy_conversation_js():
    return """
    <script>
    function copyToClipboard() {
        let chatbox = document.querySelector('.chatbot');
        let text = Array.from(chatbox.querySelectorAll('.message')).map(msg => msg.innerText).join('\\n');
        navigator.clipboard.writeText(text).then(function() {
            console.log('Text copied to clipboard');
        }).catch(function(error) {
            console.error('Error copying text: ', error);
        });
    }
    document.addEventListener('DOMContentLoaded', function() {
        let button = document.getElementById('copy-btn');
        button.addEventListener('click', copyToClipboard);
    });
    </script>
    """
       
def clear_chat(image_input, chatbot):
    image_input.clear()
    return [], ""
    
with gr.Blocks() as demo:
    gr.Markdown("# Langchain Working Bench")
    gr.Markdown("Chat with LLM's of choice and use agents to get work done.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                label="Choose Model",
                value="Ollama (LLaVA)"
            )
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            chat_bot = gr.Chatbot(height=600)
            chat_text_box = gr.Textbox(label="Chat input", placeholder="Type your message here...")
            chat_history_flag = gr.Checkbox(label="Include conversation history", value=True)
            chat_interface = gr.ChatInterface(
                fn=chat_wrapper,
                chatbot=chat_bot,
                textbox=chat_text_box,
                additional_inputs=[model_choice, chat_history_flag],
                submit_btn="Submit",
                retry_btn="🔄 Retry",
                undo_btn="↩️ Undo",
                clear_btn="🗑️ Clear",
            )

        with gr.Tab("Agent"):
            prompt_chat_bot = gr.Chatbot(height=600)
            prompt_text_box = gr.Textbox(label="Prompt input", placeholder="Type your prompt here...")
            prompt_info = gr.Dropdown(choices=get_prompt_list(), label="Agent Selection", interactive=True)
            prompt_interface = gr.ChatInterface(
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
                    history_flag = gr.Checkbox(label="Include conversation history", value=True)
                
                with gr.Column(scale=2):
                    vision_chatbot = gr.Chatbot(show_copy_button=True, height=400)
                    vision_question_input = gr.Textbox(label="Ask about the image", placeholder="Type your question about the image here...")
                    with gr.Row():
                        vision_submit_btn = gr.Button("Submit")
                        vision_copy_btn = gr.Button("Copy Conversation", elem_id="vision-copy-btn")
                        vision_clear_btn = gr.Button("Clear")
                        
    # Process image when submit button is clicked
    def handle_submit(image, question, model_choice, history, history_flag):
        if not question:
            return history, "Please enter a question."
        if not image:
            return history, "Please upload an image."

        response = process_image_wrapper(image, question, model_choice)
        history.append((question, "".join(response)))
        return history, ""
    
    # Vision Assistant event handlers
    vision_submit_btn.click(
        handle_submit,
        inputs=[image_input, vision_question_input, model_choice, vision_chatbot, history_flag],
        outputs=[vision_chatbot, vision_question_input]
    )
    
    vision_clear_btn.click(
        lambda: clear_chat(image_input, vision_chatbot),
        inputs=[],
        outputs=[vision_chatbot, vision_question_input]
    )
    
    vision_copy_btn.click(
        lambda: gr.HTML(copy_conversation_js()),
        inputs=[],
        outputs=[]
    )

    gr.HTML(copy_conversation_js())

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=False)