# Main Gradio AI workbench app
# Standard library imports
import os
import logging
import sys
import asyncio
from typing import List, Union, Tuple
from pathlib import Path
import traceback

# Third-party imports
from PIL import Image
import gradio as gr
from langchain_core.documents import Document

# Local imports
from ai_model_core.config.settings import (
    load_config, 
    get_prompt_list, 
    update_prompt_list
)
from ai_model_core.shared_utils.factory import (
    get_model,
    get_embedding_model,
    WHISPER_MODELS,
    OUTPUT_FORMATS
)
from ai_model_core.shared_utils.utils import ( 
    EnhancedContentLoader,
    get_prompt_template,
    _format_history
)
from ai_model_core.model_helpers import (
    ChatAssistant, 
    PromptAssistant, 
    VisionAssistant,
    RAGAssistant, 
    SummarizationAssistant,
    TranscriptionAssistant
)

from ai_model_core.model_helpers.transcription_assistant import (
    TranscriptionContext
)

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
chat_assistant = ChatAssistant("Ollama (LLama3.2)")
prompt_assistant = PromptAssistant("Ollama (LLama3.2)")
vision_assistant = VisionAssistant("Ollama (LLaVA)")
rag_assistant = RAGAssistant("Ollama (LLama3.2)")
summarization_assistant = SummarizationAssistant("Ollama (LLama3.2)")
transcription_assistant = TranscriptionAssistant(model_size="base")

# Wrapper function for Gradio implementation chat_assistant:
async def chat_wrapper(
    message: str,
    history: List[Tuple[str, str]],
    model_choice: str,
    temperature: float,
    max_tokens: int,
    files: List[gr.File],
    history_flag: bool,
    use_context: bool = True
) -> str:
    
    global chat_assistant
    chat_assistant.update_model(model_choice)
    
    # Generate response
    result = []
    async for chunk in chat_assistant.chat(
        message=message,
        history=history,
        history_flag=history_flag,
        stream=True,
        use_context=use_context
    ):
        result.append(chunk)
        yield ''.join(result)

# File processing handler
async def process_files_wrapper(files: List[gr.File]) -> Tuple[str, bool]:  
    return await chat_assistant.process_chat_context_files(files)

# Wrapper function for Gradio interface prompt_assistant:
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


# Wrapper function for Gradio interface vision_assistant:
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

# Wrapper function for loading documents (RAG and summarization)
def load_documents_wrapper(url_input, file_input, chunk_size, chunk_overlap):
    try:
        loader = EnhancedContentLoader(chunk_size, chunk_overlap)
        file_paths = file_input if isinstance(file_input, list) else [file_input] if file_input else None
        docs = loader.load_and_split_documents(file_paths=file_paths, urls=url_input, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return f"Successfully loaded {len(docs)} chunks of text.", docs
    except Exception as e:
        logger.error(f"Error in load_documents: {str(e)}")
        return f"An error occurred while loading documents: {str(e)}", None

# Wrapper function for Gradio interface RAG_assistant:
async def rag_wrapper(message, history, model_choice, embedding_choice,
                      chunk_size, chunk_overlap, temperature, num_similar_docs,
                      max_tokens, urls, files, language, prompt_info,
                      history_flag, retrieval_method):

    # Create the embedding model based on the choice
    embedding_model = get_embedding_model(embedding_choice)

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
        content_loader = EnhancedContentLoader(chunk_size, chunk_overlap)
        
        logger.debug(f"Type of files: {type(files)}")
        if files:
            logger.debug(f"Number of files: {len(files)}")
            logger.debug(f"Type of first file: {type(files[0])}")
            logger.debug(f"Attributes of first file: {dir(files[0])}")
            
        docs = content_loader.load_and_split_documents(
            file_paths=files, urls=urls, chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)        
        rag_assistant.setup_vectorstore(docs)
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

# Wrapper function for Gradio interface summarize_assistant:
async def summarize_wrapper(
    loaded_docs: Union[List[Document], List[str], None],
    model_choice: str,
    method: str,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
    temperature: float,
    prompt_info: str,
    language: str,
    verbose: bool
) -> str:
    """
    Wrapper function for document summarization using the SummarizationAssistant.
    
    Args:
        loaded_docs: List of Document objects or strings to summarize
        model_choice: Name of the language model to use
        method: Summarization method (handled by Gradio dropdown)
        chunk_size: Size of text chunks for processing (handled by Gradio slider)
        chunk_overlap: Overlap between consecutive chunks (handled by Gradio slider)
        max_tokens: Maximum tokens in the final summary (handled by Gradio slider)
        temperature: Temperature parameter for text generation (handled by Gradio slider)
        prompt_info: Type of summary prompt to use
        language: Language for summarization (handled by Gradio dropdown)
        verbose: Enable verbose logging (handled by Gradio checkbox)
    """
    # Only check for loaded documents as this isn't handled by Gradio UI constraints
    if loaded_docs is None or len(loaded_docs) == 0:
        return "Error: No documents loaded. Please load documents before summarizing."

    try:
        if verbose:
            logger.info(f"Starting summarization with method: {method}")
            logger.info(f"Processing {len(loaded_docs)} documents")
        
        # Initialize summarizer
        summarizer = SummarizationAssistant(
            model_name=model_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=method,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_info=prompt_info,
            language_choice=language,
            verbose=verbose
        )

        # Perform summarization
        result = await summarizer.summarize(
            chunks=loaded_docs, 
            method=method,
            prompt_info=prompt_info,
            language=language
        )
        
        if not result or 'final_summary' not in result:
            return "Error: No summary generated. Please check your inputs and try again."
        
        if verbose:
            logger.info(f"Successfully generated summary of length {len(result['final_summary'])}")
        
        return result['final_summary']

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Summarization error: {str(e)}\n{error_trace}")
        return f"An error occurred during summarization: {str(e)}"
    
# Helper functions for Gradio interface
def clear_chat():
    return None

def clear_vision_chat():
    return None, None, gr.update(value=None)
flagging_callback = gr.CSVLogger()

async def transcription_wrapper_streaming(
    media_input, url_input, model_choice, language, task_type,
    device_input, output_format, speakers_input, terms_input,
    context_input, temperature=0.0, verbose=True,
    progress=gr.Progress()
):
    if not media_input and not url_input:
        yield "Please provide input", None, None, None, None, None, None, None, "Error: No input", ""
        return

    try:
        progress(0, desc="Initializing...")
        content_loader = EnhancedContentLoader()
        audio_path = media_input

        if url_input:
            if not url_input.startswith(('http://', 'https://')):
                url_input = 'https://' + url_input.lstrip('/')
            temp_path = await asyncio.to_thread(content_loader._download_audio_file, url_input)
            if not temp_path:
                raise ValueError(f"Failed to download audio from URL: {url_input}")
            if not os.path.exists(temp_path):
                wav_path = temp_path + '.wav'
                if os.path.exists(wav_path):
                    temp_path = wav_path
            audio_path = temp_path

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Process transcription context
        context = TranscriptionContext()
        if speakers_input:
            context.speakers = [s.strip() for s in speakers_input.split(',')]
        if terms_input:
            term_pairs = [term.strip().split(':') for term in terms_input.split(',')]
            context.terms = {pair[0].strip(): pair[1].strip() 
                           for pair in term_pairs if len(pair) == 2}
        if context_input:
            context.context = context_input

        transcription_assistant = TranscriptionAssistant(
            model_size=model_choice.split()[-1].lower(),
            language=None if language == "Auto" else language,
            task_type=task_type,
            device=device_input.lower(),
            temperature=temperature,
            output_dir="./output",
            context=context,
            verbose=verbose
        )

        async def progress_callback(percent, status):
            progress(percent / 100, desc=status)

        all_segments = []
        final_text = ""

        async for chunk_result in transcription_assistant.process_audio_streaming(
            audio_path, progress_callback=progress_callback
        ):
            if isinstance(chunk_result, dict):
                final_text = chunk_result.get("current_text", "")
                raw_text = chunk_result.get("raw_text", final_text)
                if "segments" in chunk_result:
                    all_segments.extend(chunk_result["segments"])

                yield (
                    final_text,
                    None, None,
                    None, None, None, None, None,
                    chunk_result.get("status", ""),
                    chunk_result.get("processed_time", "")
                )

        if output_format != "none" and raw_text:
            stem = Path(audio_path).stem
            transcription_state = {
                'audio_path': audio_path,
                'transcription': raw_text,
                'results': {
                    "text": raw_text,
                    "segments": all_segments,
                    "language": language if language != "Auto" else "en"
                },
                'all_actions': [],
                'selected_format': output_format
            }
            
            await transcription_assistant.save_outputs(transcription_state)
            
            yield (
                final_text,
                None, None,
                f"./output/{stem}.txt" if output_format in ['txt', 'all'] else None,
                f"./output/{stem}.srt" if output_format in ['srt', 'all'] else None,
                f"./output/{stem}.vtt" if output_format in ['vtt', 'all'] else None,
                f"./output/{stem}.tsv" if output_format in ['tsv', 'all'] else None,
                f"./output/{stem}.json" if output_format in ['json', 'all'] else None,
                "Processing complete",
                chunk_result.get("processed_time", "")
            )
        else:
            yield (
                final_text,
                None, None,
                None, None, None, None, None,
                "Processing complete",
                chunk_result.get("processed_time", "")
            )

        if url_input and temp_path:
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        yield str(e), None, None, None, None, None, None, None, f"Error: {str(e)}", ""
                                                
# Function to update button state
def start_transcription(*args):
    return {
        transcribe_button: "Transcribing...",
        transcribing: "true"
    }

def finish_transcription(*args):
    return {
        transcribe_button: "Start Transcription",
        transcribing: "false"
    }

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Get work done with LLM's of choice")

    with gr.Tabs():
        # Chat Tab
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.2)", "Claude Sonnet", 
                         "Claude Sonnet beta", "Deepseek v3",
                         "Ollama (Deepseek r1)", "Mistral (large)", "Mistral (small)",
                         "Ollama (LLama3.1)", "OpenAI GPT-4o-mini"],
                        label="Choose Model",
                        value="Ollama (LLama3.2)"
                    )
                    # File upload section
                    with gr.Group():
                        file_input = gr.File(
                            label="Upload Context Content",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".py"]
                            )
                        load_button = gr.Button("Load in chat context")
                        load_output = gr.Textbox(
                        label="Load Status", interactive=False
                        )
                        use_context = gr.Checkbox(
                            label="Use uploaded files as context",
                            value=True
                            )
                        
                    with gr.Accordion("Model parameters", open=False):
                        history_flag = gr.Checkbox(
                        label="Include conversation history", value=True
                    )
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.1, step=0.1,
                            label="Temperature"
                        )
                        max_tokens = gr.Slider(
                            minimum=150, maximum=4000, value=500, step=50,
                            label="Max Tokens"
                        )
                with gr.Column(scale=4):
                    chat_bot = gr.Chatbot(
                        height=600,
                        show_copy_button=True,
                        show_copy_all_button=True
                    )
                    chat_text_box = gr.Textbox(
                        label="User input",
                        placeholder="Type your question here..."
                    )
                    
                    chat_interface = gr.ChatInterface(
                        fn=chat_wrapper,
                        chatbot=chat_bot,
                        textbox=chat_text_box,
                        additional_inputs=[
                            model_choice, temperature, max_tokens, file_input,
                            use_context, history_flag
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )

            # Connect the load_button to the process_chat_context_files
            loaded_docs = gr.State()
            load_button.click(
                fn=lambda files: asyncio.run(process_files_wrapper(files)),
                inputs=[file_input],
                outputs=[load_output, loaded_docs]
            )
            
        # Prompting Tab
        with gr.Tab("Prompting"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.2)", "Claude Sonnet",
                         "Claude Sonnet beta", "Deepseek v3",
                         "Mistral (large)", "Mistral (small)",
                         "OpenAI GPT-4o-mini", "Ollama (LLama3.1)"],
                        label="Choose Model",
                        value="Ollama (LLama3.2)"
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
                        ["Ollama (LLaVA)", "Ollama (llama3.2-vision)", 
                         "Mistral (pixtral)", "OpenAI GPT-4o-mini",
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
                        file_types=[".txt", ".md", ".pdf", ".docx"],
                        file_count="multiple"
                    )
                    with gr.Accordion("RAG Options", open=False):
                        model_choice = gr.Dropdown(
                            ["Ollama (LLama3.2)", "Claude Sonnet",
                             "Claude Sonnet beta", "Deepseek v3",
                             "Mistral (large)", "Mistral (small)",
                             "Ollama (phi3.5)", "OpenAI GPT-4o-mini"],
                            label="Choose Model",
                            value="Ollama (LLama3.2)"
                        )
                        embedding_choice = gr.Dropdown(
                            [
                                    "nomic-embed-text",
                                    "e5-base",
                                    "bge-large",
                                    "bge-m3",
                                    "e5-large",
                                    "text-embedding-ada-002",
                                    "text-embedding-3-small",
                                    "text-embedding-3-large"
                                ],
                            label="Choose Embedding Model",
                            value="nomic-embed-text"
                        )
                        chunk_size = gr.Slider(
                            minimum=100, maximum=2500, value=500, step=100,
                            label="Chunk Size"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, maximum=250, value=50, step=10,
                            label="Chunk Overlap"
                        )
                        max_tokens = gr.Slider(
                            minimum=50, maximum=4000, value=1000, step=50,
                            label="Max token generation"
                        )
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.1, step=0.1,
                            label="Temp text generation"
                        )
                        retrieval_method = gr.Dropdown(
                            choices=["similarity", "mmr",
                                     "similarity_threshold"],
                            label="Select Retriever Method",
                            value="similarity"
                        )
                        num_similar_docs = gr.Slider(
                            minimum=2, maximum=5, value=3, step=1,
                            label="Search Number of Fragments"
                        )

                    load_button = gr.Button("Load content")
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
                        height=600, 
                        show_copy_button=True,
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
                            max_tokens, url_input, file_input,
                            language_choice, prompt_info, history_flag,
                            retrieval_method
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )
                    flag_btn = gr.Button("Flag")
                    flag_options = gr.Dropdown(
                        ["High quality", "OK", "Incorrect", "Ambiguous",
                         "Inappropriate"],
                        label="Flagging Options"
                    )

            # Connect the load_button to the load_documents_wrapper function
            loaded_docs = gr.State()
            load_button.click(
                fn=load_documents_wrapper,
                inputs=[url_input, file_input, chunk_size, chunk_overlap],
                outputs=[load_output, loaded_docs]
            )

        # Summarization Assistant Tab
        with gr.Tab("Summarization Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(
                        label="Webpage to load",
                        placeholder="Enter URLs here",
                        lines=1
                    )
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".pdf", ".md", ".docx"],
                        file_count="single"
                    )
                    chunk_size = gr.Slider(
                        minimum=100, maximum=5000,
                        value=2000, step=100,
                        label="Chunk Size"
                        )
                    chunk_overlap = gr.Slider(
                        minimum=0, maximum=500,
                        value=200, step=10,
                        label="Chunk Overlap"
                        )
                    load_button = gr.Button("Load content")
                    load_output = gr.Textbox(
                        label="Load Status", interactive=False
                    )
                    with gr.Accordion("Summarization Options", open=False):
                        model_choice = gr.Dropdown(
                            ["Ollama (LLama3.2)", "Claude Sonnet",
                             "Claude Sonnet beta", "Deepseek v3",
                             "Mistral (large)", "Mistral (small)",
                             "Ollama (phi3.5)", "OpenAI GPT-4o-mini"],
                            label="Choose Model",
                            value="Ollama (LLama3.2)"
                        )
                        language_choice = gr.Dropdown(
                            ["english", "dutch"],
                            label="Choose language",
                            value="english"
                        )
                        method = gr.Dropdown(
                                ["stuff", "map_reduce", "refine"],
                                label="Summarization Strategy",
                                value="stuff"
                            )
                        max_tokens = gr.Slider(
                            minimum=50, maximum=4000,
                            value=3000, step=50,
                            label="Max Tokens"
                            )
                        temperature = gr.Slider(
                            minimum=0, maximum=1,
                            value=0.4, step=0.1,
                            label="Temperature"
                            )
                        verbose = gr.Checkbox(
                            label="Verbose Mode",
                            value=False
                            )

                with gr.Column(scale=4):
                    summary_output = gr.Textbox(
                        label="Summary Output",
                        lines=25,
                        show_copy_button=True
                    )
                    summarize_button = gr.Button("Summarize Document")

            # Connect the load_button to the load_documents_wrapper function
            loaded_docs = gr.State()
            load_button.click(
                fn=load_documents_wrapper,
                inputs=[url_input, file_input, chunk_size, chunk_overlap],
                outputs=[load_output, loaded_docs]
            )

            summarize_button.click(
                fn=summarize_wrapper,
                inputs=[
                    loaded_docs, model_choice, method, chunk_size,
                    chunk_overlap, max_tokens, temperature, 
                    language_choice, prompt_info, verbose
                ],
                outputs=summary_output
            )
            
        with gr.Tab("Transcription Assistant"):
            with gr.Row():
                # Left Column - Input Controls
                with gr.Column(scale=1):
                    media_input = gr.File(
                        label="Upload Media File",
                        file_types=["audio", "video"],
                        file_count="single",
                        type="filepath"
                    )
                    url_input = gr.Textbox(
                        label="Video URL",
                        placeholder="Enter video URL here"
                    )
                    
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=["Auto", "en", "nl", "de", "fr", "bg"],
                            value="Auto",
                            label="Source Language",
                            info="Specification improves speed"
                        )
                    
                    with gr.Accordion("Options", open=False):
                        model_choice = gr.Dropdown(
                            choices=WHISPER_MODELS,
                            value="Whisper large",
                            label="Model Size",
                            info="Larger models are more accurate but slower"
                        )
                        task_type = gr.Radio(
                            choices=["transcribe", "translate"],
                            value="transcribe",
                            label="Task Type",
                            info=(
                                "'Transcribe' keeps original language, "
                                "'Translate' converts to English"
                            )
                        )
                        output_format = gr.Dropdown(
                            choices=OUTPUT_FORMATS,
                            value="none",
                            label="Output Format",
                            info="Select output file format or 'none' for no file output"
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                            info="Higher values = more random output"
                        )
                        device_input = gr.Radio(
                            choices=["CPU", "GPU"],
                            value="CPU",
                            label="Device",
                            info="GPU support requires additional setup"
                        )
                        verbose = gr.Checkbox(
                            value=True,
                            label="Verbose Output",
                            info="Show detailed progress"
                        )

                # Right Column - Output and Progress
                with gr.Column(scale=2):
                    with gr.Accordion("Transcription Context Hints", open=False):
                        speakers_input = gr.Textbox(
                            label="Speakers",
                            placeholder="Geoffrey Hinton, Andrej Karpathy",
                            info="Add speaker names for correct spelling, separated by commas"
                        )
                        context_input = gr.Textbox(
                            label="Additional Context",
                            placeholder="Meeting about customer implementation",
                            lines=2,
                            info="Add brief context to help with domain-specific transcription"
                        )
                        terms_input = gr.Textbox(
                            label="Specialized terms",
                            placeholder="LLM:Large Language Model, RAG:Retrieval Augmented Generation",
                            info="Add technical terms as term:description, separated by commas"
                        )
                    # Always visible component
                    subtitle_preview = gr.TextArea(
                        label="Transcription Preview",
                        interactive=False,
                        show_copy_button=True
                    )
                    
                    # Progress info in collapsible accordion
                    with gr.Accordion("Processing Details", open=False):
                        progress_bar = gr.Progress()
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        time_info = gr.Textbox(
                            label="Processing Time",
                            interactive=False
                        )
                    
                    # Hidden video/audio outputs
                    video_output = gr.Video(
                        label="Transcribed Video",
                        visible=False
                    )
                    audio_output = gr.Audio(
                        label="Transcribed Audio",
                        visible=False
                    )
                    
                    # Downloads in separate accordion
                    with gr.Accordion("Downloads", open=False):
                        txt_download = gr.File(
                            label="TXT Download",
                            visible=lambda: output_format not in ["none"]
                        )
                        srt_download = gr.File(
                            label="SRT Download",
                            visible=lambda: output_format not in ["none"]
                        )
                        vtt_download = gr.File(
                            label="VTT Download",
                            visible=lambda: output_format not in ["none"]
                        )
                        tsv_download = gr.File(
                            label="TSV Download",
                            visible=lambda: output_format not in ["none"]
                        )
                        json_download = gr.File(
                            label="JSON Download",
                            visible=lambda: output_format not in ["none"]
                        )

            # Process button under both columns
            transcribe_button = gr.Button(
                "Start Transcription",
                variant="primary",
                elem_classes=["primary-btn"]
            )
            
            # Add interactive elements for button state
            transcribing = gr.Textbox(value="", visible=False)

            # Connect the transcribe button to the wrapper function
            transcribe_button.click(
                fn=start_transcription,
                outputs=[transcribe_button, transcribing]
            ).then(
                fn=transcription_wrapper_streaming,
                inputs=[
                    media_input, url_input, model_choice, language,
                    task_type, device_input, output_format, 
                    speakers_input, terms_input, context_input,
                    temperature, verbose
                ],
                outputs=[
                    subtitle_preview,
                    audio_output, 
                    video_output,
                    txt_download, 
                    srt_download, 
                    vtt_download,
                    tsv_download, 
                    json_download,
                    status_text,
                    time_info
                ],
                show_progress="full"
            ).then(
                fn=finish_transcription,
                outputs=[transcribe_button, transcribing]
            )

    # Set up the flagging callback
    flagging_callback.setup(
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs,
        "flagged_rag_data"
    )

    # Connect the flagging button to the callback
    flag_btn.click(
        lambda *args: flagging_callback.flag(args[:-1] + (args[-1],)),
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs +
        [flag_options],
        None,
        preprocess=False
    )

    language_choice.change(
        fn=update_prompt_list,
        inputs=[language_choice],
        outputs=[prompt_info]
    )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface of the ai-workbench")
    demo.launch(server_port=7860, debug=True, share=False)