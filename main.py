import os
import logging
import sys
from typing import List, Union
from PIL import Image
import gradio as gr
from ai_model_core.config.settings import (
    load_config, get_prompt_list, update_prompt_list
)
from ai_model_core.model_helpers import (
    ChatAssistant, PromptAssistant, VisionAssistant,
    RAGAssistant, SummarizationAssistant
)
from ai_model_core.model_helpers.RAG_assistant import (
    CustomHuggingFaceEmbeddings
)
from ai_model_core import get_embedding_model

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['USER_AGENT'] = 'my-RAG-agent'

# Load config at startup
config = load_config()

# Set up logging
DEBUG_MODE = config.get('debug_mode', False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('app.log')
c_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
f_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.propagate = False

# Initialize assistants with default models
chat_assistant = ChatAssistant("Ollama (LLama3.1)")
prompt_assistant = PromptAssistant("Ollama (LLama3.1)")
vision_assistant = VisionAssistant("Ollama (LLaVA)")
rag_assistant = RAGAssistant("Ollama (LLama3.1)")
summarization_assistant = SummarizationAssistant("Ollama (LLama3.1)")

# Wrapper functions for Gradio interface
async def chat_wrapper(message, history, model_choice, history_flag,
                       temperature, max_tokens):
    global chat_assistant
    chat_assistant.update_model(model_choice)
    chat_assistant.temperature = temperature
    chat_assistant.max_tokens = max_tokens

    try:
        result = []
        async for chunk in chat_assistant.chat(
            message, history, history_flag, stream=True
        ):
            result.append(chunk)
            yield ''.join(result)
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        yield f"An error occurred: {str(e)}"

async def prompt_wrapper(
    message: str,
    history: List[tuple[str, str]],
    model_choice: str,
    prompt_info: str,
    language_choice: str,
    history_flag: bool,
    stream: bool = False
):
    global prompt_assistant
    prompt_assistant.update_model(model_choice)

    result = []
    async for chunk in prompt_assistant.prompt(
        message, history, prompt_info, language_choice, history_flag, stream
    ):
        result.append(chunk)
        yield ''.join(result)

async def process_image_wrapper(
    message: str,
    history: List[tuple[str, str]],
    image: Union[Image.Image, str, None],
    model_choice: str,
    history_flag: bool,
    stream: bool = False
):
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

def load_content(url_input, file_input, model_choice, embedding_choice,
                 chunk_size, chunk_overlap, max_tokens):
    try:
        global rag_assistant
        if not hasattr(globals(), 'rag_assistant') or rag_assistant is None:
            rag_assistant = RAGAssistant(
                model_name=model_choice,
                embedding_model=embedding_choice,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_tokens=max_tokens
            )
        
        rag_assistant.setup_vectorstore(url_input, file_input)
        return "Content loaded successfully into memory."
    except Exception as e:
        return f"Error loading content: {str(e)}"

async def rag_wrapper(message, history, model_choice, embedding_choice,
                      chunk_size, chunk_overlap, temperature, num_similar_docs,
                      max_tokens, urls, files, language, prompt_info,
                      history_flag, retrieval_method):
    embedding_model = (
        CustomHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        if embedding_choice == "all-MiniLM-L6-v2"
        else get_embedding_model(embedding_choice)
    )
    
    rag_assistant = RAGAssistant(
        model_name=model_choice,
        embedding_model=embedding_choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
        num_similar_docs=num_similar_docs,
        language=language,
        max_tokens=max_tokens
    )
    
    try:
        logger.info("Setting up vectorstore")
        rag_assistant.setup_vectorstore(urls, files, retrieval_method)
        rag_assistant.prompt_template = prompt_info
        rag_assistant.use_history = history_flag
        
        logger.info("Querying RAG assistant")
        result = await rag_assistant.query(
            message, history if history_flag else None
        )
        return result
    except Exception as e:
        logger.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred: {str(e)}"

async def summarize_wrapper(file_input, model_choice, chain_type, chunk_size,
                            chunk_overlap, max_tokens, temperature, language,
                            verbose):
    summarizer = SummarizationAssistant(
        model_name=model_choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_tokens=max_tokens,
        temperature=temperature,
        chain_type=chain_type,
        language=language,
        verbose=verbose
    )

    if not file_input:
        return "Please upload a file to summarize."

    try:
        summary = await summarizer.summarize(file_input.name)
        return summary
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

# Helper functions for Gradio interface
def clear_chat():
    return None

def clear_vision_chat():
    return None, None, gr.update(value=None)

flagging_callback = gr.CSVLogger()

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Chat with LLM's of choice and reuse prompts to get work done.")

    with gr.Tabs():
        # Chat Tab
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Claude Sonnet",
                         "Ollama (Deepseek-coder-v2)",
                         "Ollama (YI-coder)", "OpenAI GPT-4o-mini"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    history_flag = gr.Checkbox(
                        label="Include conversation history", value=True
                    )
                    with gr.Accordion("Model parameters", open=False):
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        max_tokens = gr.Slider(
                            minimum=150, maximum=4000, value=500, step=50,
                            label="Max Tokens"
                        )
                with gr.Column(scale=4):
                    chat_bot = gr.Chatbot(
                        height=600, show_copy_button=True,
                        show_copy_all_button=True
                    )
                    chat_text_box = gr.Textbox(
                        label="Chat input",
                        placeholder="Type your message here..."
                    )
                    gr.ChatInterface(
                        fn=chat_wrapper,
                        chatbot=chat_bot,
                        textbox=chat_text_box,
                        additional_inputs=[
                            model_choice, history_flag, temperature, max_tokens
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

        # Prompting Tab
        with gr.Tab("Prompting"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)",  "Ollama (phi3.5)",
                         "OpenAI GPT-4o-mini", "Claude Sonnet"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Language",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(
                        choices=get_prompt_list(language_choice.value),
                        label="Prompt Selection", interactive=True
                    )
                    history_flag_prompt = gr.Checkbox(
                        label="Include conversation history", value=True
                    )

                with gr.Column(scale=4):
                    prompt_chat_bot = gr.Chatbot(
                        height=600, show_copy_button=True,
                        show_copy_all_button=True
                    )
                    prompt_text_box = gr.Textbox(
                        label="Prompt input",
                        placeholder="Type your prompt here..."
                    )
                    gr.ChatInterface(
                        fn=prompt_wrapper,
                        chatbot=prompt_chat_bot,
                        textbox=prompt_text_box,
                        additional_inputs=[
                            model_choice, prompt_info, language_choice,
                            history_flag_prompt
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

        # Vision Assistant Tab
        with gr.Tab("Vision Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", label="Upload Image", image_mode="RGB"
                    )
                    model_choice = gr.Dropdown(
                        ["Ollama (LLaVA)", "OpenAI GPT-4o-mini",
                         "Claude Sonnet"],
                        label="Choose Model",
                        value="Ollama (LLaVA)"
                    )
                    history_flag = gr.Checkbox(
                        label="Include conversation history", value=True
                    )

                with gr.Column(scale=4):
                    vision_chatbot = gr.Chatbot(
                        height=600, show_copy_button=True,
                        show_copy_all_button=True
                    )
                    vision_question_input = gr.Textbox(
                        label="Ask about the image",
                        placeholder="Type your question about the image here..."
                    )
                    gr.ChatInterface(
                        fn=process_image_wrapper,
                        chatbot=vision_chatbot,
                        textbox=vision_question_input,
                        additional_inputs=[
                            image_input, model_choice, history_flag
                        ],
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

        # RAG Assistant Tab
        with gr.Tab("RAG Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(
                        label="Webpages to load (one per line)",
                        placeholder="Enter URLs here, one per line",
                        lines=3
                    )
                    file_input = gr.File(
                        label="Upload Text Documents",
                        file_types=[".txt", ".pdf", ".docx"],
                        file_count="multiple"
                    )
                    with gr.Accordion("RAG Options", open=False):
                        model_choice = gr.Dropdown(
                            ["Ollama (LLama3.1)", "Claude Sonnet",
                             "Ollama (phi3.5)", "OpenAI GPT-4o-mini"],
                            label="Choose Model",
                            value="Ollama (LLama3.1)"
                        )
                        embedding_choice = gr.Dropdown(
                            ["nomic-embed-text", "all-MiniLM-L6-v2",
                             "text-embedding-ada-002"],
                            label="Choose Embedding Model",
                            value="nomic-embed-text"
                        )
                        max_tokens = gr.Slider(
                            minimum=50, maximum=4000, value=1000, step=50,
                            label="Max token generation"
                        )
                        chunk_size = gr.Slider(
                            minimum=100, maximum=2500, value=500, step=100,
                            label="Fragment Size"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, maximum=250, value=50, step=10,
                            label="Fragment Overlap"
                        )
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.1, step=0.1,
                            label="Temp text generation"
                        )
                        retrieval_method = gr.Dropdown(
                            choices=["similarity", "mmr", "similarity_threshold"],
                            label="Select Retriever Method", value="similarity"
                        )
                        num_similar_docs = gr.Slider(
                            minimum=2, maximum=10, value=3, step=1,
                            label="Search Number of Fragments"
                        )

                    load_button = gr.Button("Process content for analysis")
                    load_output = gr.Textbox(
                        label="Load Status", interactive=False
                    )

                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Prompt Family",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(
                        choices=get_prompt_list(language_choice.value),
                        label="Prompt Template", interactive=True
                    )
                    history_flag = gr.Checkbox(
                        label="Include conversation history", value=True
                    )
                    
                with gr.Column(scale=4):
                    rag_chat_bot = gr.Chatbot(
                        height=600, show_copy_button=True,
                        show_copy_all_button=True
                    )
                    rag_text_box = gr.Textbox(
                        label="RAG input",
                        placeholder="Type your question here..."
                    )
                    chat_interface = gr.ChatInterface(
                        fn=rag_wrapper,
                        chatbot=rag_chat_bot,
                        textbox=rag_text_box,
                        additional_inputs=[
                            model_choice, embedding_choice, chunk_size,
                            chunk_overlap, temperature, num_similar_docs,
                            max_tokens, url_input, file_input, language_choice,
                            prompt_info, history_flag, retrieval_method
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )
                    flag_btn = gr.Button("Flag")
                    flag_options = gr.Dropdown(
                        ["High quality", "Incorrect", "Ambiguous", "Inappropriate"],
                        label="Flagging Options"
                    )

        # Summarization Assistant Tab
        with gr.Tab("Summarization Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".pdf", ".docx"],
                    )
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.1)", "Claude Sonnet",
                         "Ollama (phi3.5)", "OpenAI GPT-4o-mini"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
                    )
                    chain_type = gr.Dropdown(
                        ["stuff", "map_reduce", "refine"],
                        label="Summarization Strategy",
                        value="stuff"
                    )
                    with gr.Accordion("Summarization Options", open=False):
                        chunk_size = gr.Slider(
                            minimum=100, maximum=5000,
                            value=500, step=100,
                            label="Chunk Size"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, maximum=500,
                            value=200, step=10,
                            label="Chunk Overlap"
                        )
                        max_tokens = gr.Slider(
                            minimum=50, maximum=4000,
                            value=1000, step=50,
                            label="Max Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0, maximum=1,
                            value=0.4, step=0.1,
                            label="Temperature"
                        )
                        language = gr.Dropdown(
                            ["english", "dutch"],
                            label="Choose Language",
                            value="english"
                        )
                        verbose = gr.Checkbox(
                            label="Verbose Mode",
                            value=False
                        )

                with gr.Column(scale=4):
                    summary_output = gr.Textbox(
                        label="Summary Output",
                        lines=10
                    )
                    summarize_button = gr.Button("Summarize Document")

            summarize_button.click(
                fn=summarize_wrapper,
                inputs=[
                    file_input, model_choice, chain_type, chunk_size,
                    chunk_overlap, max_tokens, temperature, language,
                    verbose
                ],
                outputs=summary_output
            )

    # Set up the flagging callback
    flagging_callback.setup(
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs,
        "flagged_rag_data"
    )

    # Connect the flagging button to the callback
    flag_btn.click(
        lambda *args: flagging_callback.flag(args[:-1] + (args[-1],)),
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs + [flag_options],
        None,
        preprocess=False
    )

    load_button.click(
        fn=load_content,
        inputs=[
            url_input, file_input, model_choice, embedding_choice,
            chunk_size, chunk_overlap, max_tokens
        ],
        outputs=load_output
    )

    language_choice.change(
        fn=update_prompt_list,
        inputs=[language_choice],
        outputs=[prompt_info]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True, share=False)