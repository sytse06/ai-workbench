import logging
import os
#import json
import yaml
#from functools import partial
from io import BytesIO
import base64
#import sys
from PIL import Image
import gradio as gr
import asyncio
from typing import List, Union, Any
from langchain_community.chat_models import ChatAnthropic, ChatOllama
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
#from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_core import get_model, get_prompt_template, get_system_prompt, _format_history
from ai_model_core.model_helpers import ChatAssistant, PromptAssistant, VisionAssistant, RAGAssistant

# Load config at startup
config = load_config()

# Set up logging
DEBUG_MODE = config.get('debug_mode', False)
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize assistants with default models
chat_assistant = ChatAssistant("Ollama (LLama3.1)")
prompt_assistant = PromptAssistant("Ollama (LLama3.1)")
vision_assistant = VisionAssistant("Ollama (LLaVA)")
rag_assistant = RAGAssistant("Ollama (LLama3.1)")

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

async def prompt_wrapper(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, language_choice: str, history_flag: bool, stream: bool = False):
    global prompt_assistant
    
    prompt_assistant.update_model(model_choice)

    result = []
    async for chunk in prompt_assistant.prompt(message, history, prompt_info, language_choice, history_flag, stream):
        result.append(chunk)
        yield ''.join(result)

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
    
async def rag_wrapper(message, history, model_choice, embedding_choice, chunk_size, chunk_overlap, temperature, num_similar_docs):
    rag_assistant.model_local = get_model(model_choice)
    rag_assistant.embedding_model = embedding_choice
    rag_assistant.chunk_size = chunk_size
    rag_assistant.chunk_overlap = chunk_overlap
    rag_assistant.temperature = temperature
    rag_assistant.num_similar_docs = num_similar_docs

    try:
        result = await rag_assistant.query(message)
        return result
    except Exception as e:
        logger.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred: {str(e)}"
           
def clear_chat():
    return None

def clear_vision_chat():
    return None, None, gr.update(value=None)
    
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
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
                    chat_bot = gr.Chatbot(height=600, show_copy_button=True, show_copy_all_button=True)
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
                    prompt_chat_bot = gr.Chatbot(height=600, show_copy_button=True, show_copy_all_button=True)
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
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini", "Ollama (llava:7b-v1.6)"],
                        label="Choose Model",
                        value="Ollama (LLaVA)"
                    )
                    history_flag = gr.Checkbox(label="Include conversation history", value=True)

                with gr.Column(scale=4):
                    vision_chatbot = gr.Chatbot(height=600, show_copy_button=True, show_copy_all_button=True)
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
            
        with gr.Tab("RAG Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Ollama (phi3.5)", "OpenAI GPT-4o-mini", "Anthropic Claude"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                with gr.Accordion("Advanced Options", open=False):
                    embedding_choice = gr.Dropdown(
                        ["nomic-embed-text", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
                        label="Choose Embedding Model",
                        value="nomic-embed-text"
                    )
                        chunk_size = gr.Slider(minimum=100, maximum=2500, value=1000, step=100, label="Chunk Size")
                        chunk_overlap = gr.Slider(minimum=0, maximum=250, value=50, step=10, label="Chunk Overlap")
                        temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.1, label="Temperature")
                        num_similar_docs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Similar Documents")
                    url_input = gr.Textbox(
                        label="URLs to load (one per line)",
                        placeholder="Enter URLs here, one per line",
                        lines=3
                    )
                    file_input = gr.File(
                        label="Upload Text Documents",
                        file_types=[".txt", ".pdf", ".docx"],
                        multiple=True
                    )
                    load_button = gr.Button("Load URLs")
                    load_output = gr.Textbox(label="Load Status", interactive=False)
                    
                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Language",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(choices=get_prompt_list(language_choice.value), label="Prompt Selection", interactive=True)
                    history_flag = gr.Checkbox(label="Include conversation history", value=True)
                    
                with gr.Column(scale=4):
                    rag_chat_bot = gr.Chatbot(height=600, show_copy_button=True, show_copy_all_button=True)
                    rag_text_box = gr.Textbox(label="RAG input", placeholder="Type your message here...")
                    gr.ChatInterface(
                        fn=rag_wrapper,
                        chatbot=rag_chat_bot,
                        textbox=rag_text_box,
                        additional_inputs=[model_choice, embedding_choice, chunk_size, chunk_overlap, temperature, num_similar_docs, prompt_info, language_choice, history_flag],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

                 # Add flagging to the chat interface
                    chat_interface.flagging_callback = gr.FlaggingCallback(
                        ingredients=["message", "response", "embedding_choice", "chunk_size", "chunk_overlap", "temperature", "num_similar_docs"],
                        title="Flag this response",
                        description="Flag this response if it's inappropriate or incorrect.",
                        flagging_options=["Incorrect Answer", "Too small a context", "Irrelevant Retrieval", "Other Issue"]
                )                   
                                        
    load_button.click(
        fn=load_content,
        inputs=[url_input, file_input, model_choice, embedding_choice, chunk_size, chunk_overlap],
        outputs=load_output
            )

    language_choice.change(fn=update_prompt_list, inputs=[language_choice], outputs=[prompt_info])

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=DEBUG_MODE)