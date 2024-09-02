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
from langchain.prompts import ChatPromptTemplate
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from ai_model_interface.config.credentials import get_api_key, load_credentials
from ai_model_interface.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_interface import get_model, get_prompt_template, get_system_prompt, _format_history
from ai_model_interface import VisionAssistant, PromptAssistant 

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

load_credentials()
config = load_config()

async def chat(message: str, history: List[tuple[str, str]], model_choice: str, history_flag: bool, stream: bool = False):
    logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {model_choice}")
    model = get_model(model_choice)
    logger.info(f"Model instantiated: {model}")
    
    messages = []
    if history_flag:
        for human, ai in history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=message))
    
    if stream:
        return model.stream(messages)  # This returns a regular generator
    else:
        return await model.agenerate([messages])

async def prompt(formatted_prompt: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, stream: bool = False):
    logger.info(f"Formatting prompt with prompt_info: {prompt_info}")
    model = get_model(model_choice)
    system_prompt = get_prompt_list(prompt_info)
    logger.info(f"Model instantiated: {model}, system_prompt: {system_prompt}")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_prompt)
    ]
    
    if stream:
        result = []
        async for chunk in model.astream(messages, history=history):
            result.append(chunk.content)
        return result
    else:
        result = await model.agenerate([messages], history=history)
        return [result.generations[0][0].message.content]

async def process_image(image: Image.Image, question: str, model_choice: str, stream: bool = False):
    logger.info(f"Process image called with question: {question}, model_choice: {model_choice}")
    if image is None:
        return "Please upload an image first."
    
    vision_assistant = VisionAssistant(model_choice)
    logger.info(f"VisionAssistant instantiated with model_choice: {model_choice}")
    
    result = []
    async for chunk in vision_assistant.image_chat(image, question, stream=stream):
        result.append(chunk)
    
    return result

# Wrapping async functions for Gradio
async def chat_wrapper(message, history, model_choice, history_flag):
    try:
        result = await chat(message, history, model_choice, history_flag, stream=True)
        
        # Handle the regular generator in an asynchronous way
        contents = []
        for chunk in result:
            if hasattr(chunk, 'content'):
                contents.append(chunk.content)
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
        
        return ''.join(contents)
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"

async def prompt_wrapper(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, language_choice: str, history_flag: bool, stream: bool = False):
    config = load_config()
    prompt_template = get_prompt_template(prompt_info, config)
    system_prompt = get_system_prompt(language_choice, config)

    # Format the user message with the chosen prompt template
    formatted_prompt = prompt_template.format(prompt_info=prompt_info, user_message=message)

    # Handle the history if history_flag is enabled
    messages = [SystemMessage(content=system_prompt)]
    
    if history_flag:
        formatted_history = _format_history(history)
        messages.extend(formatted_history)
    
    # Add the current formatted prompt
    messages.append(HumanMessage(content=formatted_prompt))
    
    # Call the model with the full message history
    model = get_model(model_choice)
    
    if stream:  # Add this check to handle streaming
        result = []
        async for chunk in model.astream(messages):
            result.append(chunk.content)
        return ''.join(result)  # Return the concatenated result
    else:
        result = await model.agenerate([messages])
        return result.generations[0][0].message.content

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
                        ["Ollama (LLama3.1)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
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
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
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