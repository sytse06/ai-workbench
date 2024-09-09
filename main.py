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
from langchain_community.chat_models import ChatAnthropic, ChatOllama
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_core import get_model, get_prompt_template, get_system_prompt, _format_history
from ai_model_core.model_helpers import ChatAssistant, PromptAssistant, VisionAssistant 

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

vision_assistant = VisionAssistant("Ollama (LLaVA)")

async def process_image_wrapper(message: str, history: List[tuple[str, str]], image: Union[Image.Image, str, None], model_choice: str, history_flag: bool, stream: bool = False):
    if image is None or not isinstance(image, Image.Image):
        return "Please upload a valid image first."

    vision_assistant.update_model(model_choice)

    try:
        result = await vision_assistant.process_image(image, message, stream)
        
        if isinstance(result, list):
            result_text = ''.join(result)
        else:
            result_text = result

        if history_flag:
            history.append((message, result_text))
        
        return result_text
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Error in process_image_wrapper: {e}")
        logger.error("Full traceback:", exc_info=True)
        return error_message

#Copy entire conversation to clipboard        
def conversation_wrapper(assistant_type, model_choice, chat_history):
    formatted_history = ""
    if assistant_type == "chat":
        formatted_history = chat_assistant.format_conversation_history(chat_history)
    elif assistant_type == "prompt":
        formatted_history = prompt_assistant.format_conversation_history(chat_history)
    elif assistant_type == "vision":
        formatted_history = vision_assistant.format_conversation_history(chat_history)
    
    return ""
        
def copy_conversation_js():
    return """
    <script>
    function copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        console.log('Copying to clipboard was successful!');
    }

    function setupCopyButton(buttonId, wrapperName) {
        document.getElementById(buttonId).addEventListener('click', function() {
            const wrapper = gradio(wrapperName);
            wrapper.then(function(result) {
                if (result && result.data) {
                    copyToClipboard(result.data[0]);
                } else {
                    console.error('No data received from the wrapper function');
                }
            }).catch(function(err) {
                console.error('Error calling the wrapper function:', err);
            });
        });
    }

    setupCopyButton("chat-copy-btn", "conversation_wrapper");
    setupCopyButton("prompt-copy-btn", "conversation_wrapper");
    setupCopyButton("vision-copy-btn", "conversation_wrapper");
    </script>
    """
           
def clear_chat():
    return None

def clear_vision_chat():
    return None, None, gr.update(value=None)
    
with gr.Blocks() as demo:
    gr.Markdown("# AI Working Bench")
    gr.Markdown("### Chat with LLM's of choice and reuse prompts to get work done.")

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Ollama (Deepseek-coder-v2)", "Ollama (YI-coder)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
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
                    chat_copy_btn = gr.Button("Copy Chat Conversation", elem_id="chat-copy-btn")

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
                    prompt_copy_btn = gr.Button("Copy Prompt Conversation", elem_id="prompt-copy-btn")

        with gr.Tab("Vision Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB")
                    model_choice = gr.Dropdown(
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Ollama (llava:7b-v1.6)"],
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
                    vision_copy_btn = gr.Button("Copy Vision Conversation", elem_id="vision-copy-btn")

            vision_clear_btn = gr.Button("Clear All")
            vision_clear_btn.click(
                fn=clear_vision_chat,
                inputs=[],
                outputs=[vision_chatbot, vision_question_input, image_input]
            )

    language_choice.change(fn=update_prompt_list, inputs=[language_choice], outputs=[prompt_info])

    gr.HTML(copy_conversation_js())

    chat_copy_btn.click(
        fn=conversation_wrapper,
        inputs=[gr.Textbox(value="chat", visible=False), model_choice, chat_bot],
        outputs=gr.Textbox(visible=False)
    )

    prompt_copy_btn.click(
        fn=conversation_wrapper,
        inputs=[gr.Textbox(value="prompt", visible=False), model_choice, prompt_chat_bot],
        outputs=gr.Textbox(visible=False)
    )

    vision_copy_btn.click(
        fn=conversation_wrapper,
        inputs=[gr.Textbox(value="vision", visible=False), model_choice, vision_chatbot],
        outputs=gr.Textbox(visible=False)
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True)