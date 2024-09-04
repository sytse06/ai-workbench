import logging
import os
import json
import yaml
from functools import partial
from io import BytesIO
import base64
import sys
from PIL import Image
import gradio as gr
import asyncio
from typing import List, Union, Any
from langchain_community.chat_models import ChatAnthropic, ChatOllama, ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from ai_model_interface.config.credentials import get_api_key, load_credentials
from ai_model_interface.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_interface import get_model, get_prompt_template, get_system_prompt, _format_history
from ai_model_interface.model_helpers import ChatAssistant, PromptAssistant, VisionAssistant 

#print(sys.path)

# Set up logging to only show warnings and errors
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials and config for directory and prompt settings
def load_config() -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

# Initialize the ChatAssistant with a default model at the module level
chat_assistant = ChatAssistant("Ollama (LLama3.1)")

# Replace the existing chat_wrapper function with this:
async def chat_wrapper(message, history, model_choice, history_flag):
    global chat_assistant
    
    chat_assistant.update_model(model_choice)

    try:
        result = []
        async for chunk in chat_assistant.chat(message, history, history_flag, stream=True):
            result.append(chunk)
            yield ''.join(result)
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        yield f"An error occurred: {str(e)}"

# Initialize the PromptAssistant with a default model at the module level
prompt_assistant = PromptAssistant("Ollama (LLama3.1)")

async def prompt_wrapper(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, language_choice: str, history_flag: bool, stream: bool = False):
    global prompt_assistant
    
    prompt_assistant.update_model(model_choice)

    result = []
    async for chunk in prompt_assistant.prompt(message, history, prompt_info, language_choice, history_flag, stream):
        result.append(chunk)
        yield ''.join(result)

# Initialize the VisionAssistant with a default model at the module level
vision_assistant = VisionAssistant("Ollama (LLaVA)")

async def process_image_wrapper(image: Union[str, Image.Image, bytes], message: str, history: List[tuple[str, str]], model_choice: str, history_flag: bool, stream: bool = False):
    global vision_assistant 
      
    if image is None:
        yield "Please upload an image first."
        return

    vision_assistant.update_model(model_choice)
    
    result = []
    try:
        async for chunk in vision_assistant.process_image(image, message, model_choice, stream=True):
            result.append(chunk)
            yield ''.join(result)
    except Exception as e:
        yield f"An error occurred: {str(e)}"
        import traceback
        print(traceback.format_exc())

def conversation_wrapper(user_input, model_choice, chat_history_flag):
    # Get the conversation history and formatted history from your model instance
    conversation_history = model_choice.get_conversation_history()
    
    if chat_history_flag.value:
        formatted_history = model_choice._format_messages(conversation_history)
        
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
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Ollama (Deepseek-coder-v2)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    history_flag = gr.Checkbox(label="Include conversation history", value=True)
                with gr.Column(scale=4):
                    chat_bot = gr.Chatbot(height=600, show_copy_button=True)
                    chat_text_box = gr.Textbox(label="Chat input", placeholder="Type your message here...")
                    gr.ChatInterface(
                        fn=chat_wrapper,
                        chatbot=chat_bot,
                        textbox=chat_text_box,
                        additional_inputs=[model_choice, history_flag],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

        with gr.Tab("Prompting"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)",  "Ollama (phi3.5)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Language",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(choices=get_prompt_list(language_choice.value), label="Prompt Selection", interactive=True)
                    history_flag_prompt = gr.Checkbox(label="Include conversation history", value=True)

                with gr.Column(scale=4):
                    prompt_chat_bot = gr.Chatbot(height=600, show_copy_button=True)
                    prompt_text_box = gr.Textbox(label="Prompt input", placeholder="Type your prompt here...")
                    gr.ChatInterface(
                        fn=prompt_wrapper,
                        chatbot=prompt_chat_bot,
                        textbox=prompt_text_box,
                        additional_inputs=[model_choice, prompt_info, language_choice, history_flag],
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
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini"],
                        label="Choose Model",
                        value="Ollama (LLaVA)"
                    )
                    history_flag = gr.Checkbox(label="Include conversation history", value=True)

                with gr.Column(scale=4):
                    vision_chatbot = gr.Chatbot(height=600, show_copy_button=True)
                    vision_question_input = gr.Textbox(label="Ask about the image", placeholder="Type your question about the image here...")
                    gr.ChatInterface(
                        fn=process_image_wrapper,
                        chatbot=vision_chatbot,
                        textbox=vision_question_input,
                        additional_inputs=[image_input, model_choice, history_flag],
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
        inputs=[chat_text_box, model_choice, history_flag],
        outputs=[]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True)