# Main Gradio AI workbench app
# Standard library imports
import os
import datetime
import logging
import sys
import asyncio
from typing import (
    List,
    Generator,
    Any,
    Optional,
    Union,
    Tuple,
    Dict,
    Generator, 
    AsyncGenerator
)
from pathlib import Path

# Third-party imports
import gradio as gr
from langchain_core.documents import Document

# Local imports
from ai_model_core.model_helpers import (
    BaseAssistantUI,
    ChatAssistantUI,
    ChatAssistant,
    RAGAssistantUI, 
    RAGAssistant, 
    SummarizationAssistant,
    TranscriptionAssistant,
    TranscriptionContext
)
from ai_model_core.shared_utils.utils import (
    EnhancedContentLoader,
    ContentProcessingComponent,
    AssistantType,
    LoaderConfig
)
from ai_model_core.shared_utils.message_processing import MessageProcessor
from ai_model_core.shared_utils.message_types import (
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from ai_model_core.shared_utils import (
    get_model, 
    get_embedding_model, 
    update_model,
    #get_reranker,
    ModelType,
    WHISPER_MODELS, 
    OUTPUT_FORMATS
)
from ai_model_core.shared_utils.prompt_utils import (
    get_prompt_list, 
    update_prompt_list
)
from ai_model_core.config.settings import load_config

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

# Initialize document loading component during app startup
def setup_content_processing(app_config: dict) -> ContentProcessingComponent:
    """
    Initialize and configure the content processing component.
    Handles various types of content including documents, images, and audio for multimodal support.
    """
    processor = ContentProcessingComponent()
    
    # Configure RAG processing
    rag_config = LoaderConfig(
        chunk_size=app_config.get("rag", {}).get("chunk_size", 500),
        chunk_overlap=app_config.get("rag", {}).get("chunk_overlap", 50),
        process_callback=None  # Will be set after RAG assistant initialization
    )
    processor.register_assistant(AssistantType.RAG, rag_config)
    
    # Configure Summarization processing
    summary_config = LoaderConfig(
        chunk_size=app_config.get("summarization", {}).get("chunk_size", 1000),
        chunk_overlap=app_config.get("summarization", {}).get("chunk_overlap", 100),
        process_callback=None  # Will be set after Summarization assistant initialization
    )
    processor.register_assistant(AssistantType.SUMMARIZATION, summary_config)
    
    # Initialize the content loader with config settings
    processor.initialize_loader(
        temp_dir=app_config.get("directories", {}).get("temp", "input/tmp"),
        audio_sample_rate=app_config.get("audio", {}).get("sample_rate", 16000),
        perform_ocr=app_config.get("ocr", {}).get("enabled", True)
    )
    
    return processor

# Initialize assistants with default models
chat_assistant = ChatAssistant(
    model_choice=model_choice,
    temperature=temperature,
    max_tokens=max_tokens
)
chat_ui = ChatAssistantUI(chat_assistant)
rag_assistant = RAGAssistant(
    model_name="Ollama (llama3.2)",
    embedding_model="nomic-embed-text",
    temperature=0.4,
    max_tokens=1000
)
rag_assistant.ui = RAGAssistantUI(rag_assistant)
assistant_type_state = gr.State(AssistantType.RAG)
assistant_type_state = gr.State(AssistantType.SUMMARIZATION)
summarization_assistant = SummarizationAssistant("Ollama (llama3.2)")
transcription_assistant = TranscriptionAssistant(model_size="base")

# Wrapper function to instantiate chat_assistant:
async def chat_wrapper(
    message: Union[str, Dict, List],
    history: List[Dict],
    model_choice: str,
    temperature: float,
    max_tokens: int,
    use_context: bool,
    history_flag: bool = True,
    prompt_info: Optional[str] = None,
    language_choice: Optional[str] = None
) -> AsyncGenerator[List[Dict[str, str]], None]:
    """Wrapper for chat functionality with handling of context files."""
    try:
        # Extract files and text from multimodal message
        message_text = ""
        files = []
        
        # Process multimodal message from Gradio v5
        if isinstance(message, dict) and "text" in message:
            # This is the old format where files were separate
            message_text = message.get("text", "").strip()
            file_paths = message.get("files", [])
            
            # Process files if provided
            if file_paths and use_context:
                try:
                    content_loader = EnhancedContentLoader()
                    # Use the enhanced content loader to process files
                    context = content_loader.load_and_process_files(file_paths=file_paths)
                    
                    # Get formatted context as string
                    formatted_context = content_loader.get_formatted_context(
                        context,
                        format_type="string"
                    )
                    
                    # Current history
                    current_history = list(history) if history else []
                    
                    # Add context as system message if there's content
                    if formatted_context:
                        current_history.append({
                            "role": "system",
                            "content": "Context from uploaded files:\n" + formatted_context
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing files: {str(e)}")
                    current_history = list(history) if history else []
            else:
                current_history = list(history) if history else []
        elif isinstance(message, dict) and "content" in message:
            # This is the new Gradio v5 format
            if isinstance(message["content"], list):
                # Process multimodal content
                text_parts = []
                for item in message["content"]:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "path" in item:
                        # This is a file
                        files.append(item["path"])
                
                message_text = " ".join(text_parts)
            else:
                message_text = str(message["content"])
            
            # Current history
            current_history = list(history) if history else []
        else:
            # Plain text message
            message_text = str(message).strip()
            current_history = list(history) if history else []
        
        # Add the user message (without file content)
        current_history.append({"role": "user", "content": message_text})
        
        try:
            # Add empty assistant message that we'll update
            accumulated_response = ""
            assistant_message = {"role": "assistant", "content": accumulated_response}
            current_history.append(assistant_message)
            
            # Initialize ChatAssistantUI with a new ChatAssistant
            chat_assistant = ChatAssistant(
                model_choice=model_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            chat_ui = ChatAssistantUI(chat_assistant)
            
            # Process through UI
            async for response in chat_ui.process_gradio_chat(
                message=message_text,
                history=current_history[:-1] if history_flag else [],
                model_choice=model_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                files=files,
                use_context=use_context,
                history_flag=history_flag,
                prompt_info=prompt_info,
                language_choice=language_choice
            ):
                if isinstance(response, dict) and "content" in response:
                    accumulated_response += response["content"]
                    assistant_message["content"] = accumulated_response
                    yield current_history
                elif isinstance(response, str):
                    accumulated_response += response
                    assistant_message["content"] = accumulated_response
                    yield current_history

        except Exception as e:
            logger.error(f"Error in model response: {str(e)}")
            error_msg = (
                "Er is een fout opgetreden. Probeer het opnieuw."
                if language_choice and language_choice.lower() == "dutch"
                else "An error occurred. Please try again."
            )
            current_history[-1] = {"role": "assistant", "content": error_msg}
            yield current_history

    except Exception as e:
        logger.error(f"General chat wrapper error: {str(e)}")
        yield [
            {"role": "user", "content": message_text},
            {"role": "assistant", "content": "An error occurred. Please try again."}
        ]

# Add to main.py, outside of class definitions
async def process_documents_temp_summarization(docs):
    """Temporary implementation until SummarizationAssistant is updated."""
    logger.info(f"Temporarily processing {len(docs)} documents for summarization")
    # Store documents in a global variable or pass to summarization_assistant directly
    if hasattr(summarization_assistant, 'documents'):
        summarization_assistant.documents = docs
    return docs

async def process_documents_temp_transcription(docs):
    """Temporary implementation until TranscriptionAssistant is updated."""
    logger.info(f"Temporarily processing {len(docs)} documents for transcription")
    # Store documents in a global variable for later reference
    if hasattr(transcription_assistant, 'input_files'):
        transcription_assistant.input_files = docs
    return docs
                            
# Wrapper function for loading documents (RAG and summarization)
async def load_documents_wrapper(
    url_input: str,
    file_input: Union[str, List[str]],
    chunk_size: int,
    chunk_overlap: int,
    assistant_type: Optional[AssistantType] = None
) -> Tuple[str, Optional[List[Document]]]:
    """Wrapper function for loading documents through the component."""
    try:
        # Get the content processor instance
        processor = ContentProcessingComponent()
        
        # Validate inputs
        if not url_input and not file_input:
            return "No input provided. Please provide either URLs or files to process.", None
            
        # Update configuration for the specific request
        if assistant_type in [AssistantType.RAG, AssistantType.SUMMARIZATION]:
            config = processor.get_loader_config(assistant_type)
            if config:
                config.chunk_size = chunk_size
                config.chunk_overlap = chunk_overlap
        
        # Process documents
        return await processor.process_documents(
            assistant_type=assistant_type,
            url_input=url_input,
            file_input=file_input,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    except Exception as e:
        logger.error(f"Error in load_documents_wrapper: {str(e)}")
        return f"An error occurred while loading documents: {str(e)}", None

# Wrapper function for Gradio interface RAG_assistant:
async def rag_wrapper(
    message: Union[str, Dict],
    history: List[Dict],
    model_choice: str,
    embedding_choice: str,
    chunk_size: int,
    chunk_overlap: int,
    temperature: float,
    num_similar_docs: int,
    max_tokens: int,
    urls: str,
    files: List[gr.File],
    language_choice: str,
    prompt_info: Optional[str],
    history_flag: bool,
    retrieval_method: str,
    documents: Optional[List[Document]] = None
) -> AsyncGenerator[List[Dict[str, str]], None]:
    """RAG wrapper with proper message handling between Gradio and LangChain."""
    try:
        # Initialize RAGAssistant with provided parameters
        rag_assistant = RAGAssistant(
            model_name=model_choice,
            embedding_model=embedding_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_similar_docs=num_similar_docs,
            retrieval_method=retrieval_method
        )

        # Initialize RAGAssistantUI with the RAGAssistant instance
        rag_assistant.ui = RAGAssistantUI(rag_assistant)
        
        # Handle None message
        if message is None:
            logger.warning("Received None message in rag_wrapper")
            message = ""
        
        # Extract message text and files
        message_processor = MessageProcessor()
        if isinstance(message, dict):
            message_text = message.get("text", "").strip() if message.get("text") else ""
            file_paths = message.get("files", [])
            logger.debug(f"Message dict: text={message_text[:30]}..., files={len(file_paths)}")
        else:
            message_text = str(message).strip()
            file_paths = []
            logger.debug(f"Message string: {message_text[:30]}...")

        logger.info(f"Processing RAG query: {message_text[:50]}...")
        
        # Process uploaded files if any
        if file_paths or files:
            # Process files through content processor
            processor = ContentProcessingComponent()
            status_msg, docs = await processor.process_documents(
                assistant_type=AssistantType.RAG,
                file_input=file_paths or files,
                url_input=urls if urls else None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"File processing: {status_msg}")
        
        # Format current history
        current_history = list(history) if history else []
        
        # Add user message to history
        current_history.append({"role": "user", "content": message_text})
        
        # Initialize assistant message
        accumulated_response = ""
        assistant_message = {"role": "assistant", "content": accumulated_response}
        current_history.append(assistant_message)
        
        # Update the global rag_assistant instance
        await rag_assistant.update_model(model_choice)
        rag_assistant.temperature = temperature
        rag_assistant.max_tokens = max_tokens
        rag_assistant.num_similar_docs = num_similar_docs
        rag_assistant.chunk_size = chunk_size
        rag_assistant.chunk_overlap = chunk_overlap
        rag_assistant.retrieval_method = retrieval_method
        
        # Ensure we have a UI wrapper
        if not hasattr(rag_assistant, 'ui') or rag_assistant.ui is None:
            rag_assistant.ui = RAGAssistantUI(rag_assistant)
        
        # Process message through the UI wrapper
        async for response in rag_assistant.ui.process_gradio_message(
            message=message_text,
            history=current_history if history_flag else [],
            model_choice=model_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            use_context=True,
            history_flag=history_flag,
            prompt_info=prompt_info,
            language_choice=language_choice
        ):
            if isinstance(response, dict) and "content" in response:
                accumulated_response += response["content"]
                assistant_message["content"] = accumulated_response
                yield current_history
            elif isinstance(response, str):
                accumulated_response += response
                assistant_message["content"] = accumulated_response
                yield current_history

    except Exception as e:
        logger.error(f"Error in RAG wrapper: {str(e)}")
        error_msg = (
            "Er is een fout opgetreden. Probeer het opnieuw."
            if language_choice.lower() == "dutch"
            else "An error occurred. Please try again."
        )
        if len(current_history) > 0 and current_history[-1].get("role") == "assistant":
            current_history[-1]["content"] = error_msg
        else:
            current_history.append({"role": "assistant", "content": error_msg})
        yield current_history
            
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
    if not loaded_docs and hasattr(summarization_assistant, 'documents'):
        loaded_docs = summarization_assistant.documents
        
    if not loaded_docs or len(loaded_docs) == 0:
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

        # Pass documents to the summarizer
        summarizer.documents = loaded_docs

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

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Get work done with LLM's of choice")

    # Chat Tab
    tabs = gr.TabItem("Chat")
    with tabs:
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=[
                        "Ollama (llama3.2)",
                        "Ollama (llama3.2-vision)",
                        "Ollama (phi4)",
                        "Ollama (llava)",
                        "Ollama (qwen2.5:14b)",
                        "Deepseek v3",
                        "OpenAI GPT-4o-mini",
                        "OpenAI o3-mini",
                        "Claude Sonnet",
                        "Claude Sonnet beta",
                        "Mistral (large)",
                        "Mistral (pixtral)"
                    ],
                    label="Choose Model",
                    value="Ollama (llama3.2)"
                )
                with gr.Accordion("Chat Options", open=True):
                    temperature = gr.Slider(
                        minimum=0, maximum=1, value=0.1, step=0.1,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=150, maximum=4000, value=3000, step=100,
                        label="Max Tokens"
                    )
                    use_context = gr.Checkbox(  # Add this component
                        label="Use context from files",
                        value=True
                    )
                    language_choice = gr.Dropdown(
                        ["english", "dutch"],
                        label="Choose Prompt Family",
                        value="english"
                    )
                    prompt_info = gr.Dropdown(
                        choices=get_prompt_list(language_choice.value),
                        label="Prompt Template", 
                        interactive=True
                    )
                    history_flag = gr.Checkbox(
                        label="Include history", 
                        value=True
                    )                  

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    bubble_full_width=False,
                    type="messages",
                    show_copy_button=True,
                    show_copy_all_button=True,
                    height=600,
                    avatar_images=None,
                    render_markdown=True,
                    layout="bubble"
                )
                chatbot.like(print_like_dislike, None, None, like_user_message=False)
                
                # Get supported file types from EnhancedContentLoader
                content_loader = EnhancedContentLoader()
                supported_extensions = content_loader.get_supported_extensions()

                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    file_types=supported_extensions,
                    placeholder="Enter message or upload file...",
                    show_label=False,
                    container=True,
                    scale=4,
                    sources=["upload"]
                )

                with gr.Row():
                    clear = gr.ClearButton([chatbot])

                chat_input.submit(
                    fn=chat_wrapper,
                    inputs=[
                        chat_input,
                        chatbot,
                        model_choice,
                        temperature,
                        max_tokens,
                        use_context,
                        history_flag,
                        prompt_info,
                        language_choice,
                    ],
                    outputs=chatbot
                ).then(
                    fn=lambda: "",
                    inputs=None,
                    outputs=[chat_input]
            )      
            
    # Update prompt list when language changes
    language_choice.change(
        fn=update_prompt_list,
        inputs=[language_choice],
        outputs=[prompt_info]
    )
                    
    # RAG Assistant Tab
    tabs = gr.TabItem("RAG")
    with tabs:
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
                    choices=[
                        "Ollama (llama3.2)",
                        "Ollama (phi4)",
                        "Ollama (llava)",
                        "Ollama (qwen2.5:14b)",
                        "Deepseek v3",
                        "OpenAI GPT-4o-mini",
                        "OpenAI o3-mini",
                        "Claude Sonnet",
                    ],
                        label="Choose Model",
                        value="Ollama (llama3.2)"
                    )
                    embedding_choice = gr.Dropdown(
                        choices=[
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
                    type="messages", 
                    show_copy_button=True,
                    show_copy_all_button=True
                )
                rag_text_box = gr.Textbox(
                    label="RAG input",
                    placeholder="Type your question here...",
                    submit_btn=True,
                    stop_btn=True,
                )
                chat_interface = gr.ChatInterface(
                    fn=rag_wrapper,
                    type='messages',
                    chatbot=rag_chat_bot,
                    textbox=rag_text_box,
                    additional_inputs=[
                        model_choice, 
                        embedding_choice, 
                        chunk_size,
                        chunk_overlap, 
                        temperature, 
                        num_similar_docs,
                        max_tokens, 
                        url_input, 
                        file_input,
                        language_choice, 
                        prompt_info, 
                        history_flag,
                        retrieval_method
                    ],
                    submit_btn="Submit",
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
            inputs=[
                url_input, 
                file_input, 
                chunk_size, 
                chunk_overlap,
                assistant_type_state],
            outputs=[load_output, loaded_docs]
        )
        
    # Summarization Assistant Tab
    tabs = gr.TabItem("Summarization")
    with tabs:
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
            inputs=[url_input, 
                    file_input, 
                    chunk_size, 
                    chunk_overlap,
                    assistant_type_state],
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


    # Chat Tab
    tabs = gr.TabItem("Transcription")
    with tabs:
        with gr.Row():
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
            with gr.Column(scale=4):
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
    # Load configuration
    config = load_config()
    
    # Initialize content processing
    content_processor = setup_content_processing(config.get("system", {}).get("processing", {}))
    
    # Initialize assistants
    chat_assistant = ChatAssistant("Ollama (llama3.2)")
    rag_assistant = RAGAssistant(
        model_name="Ollama (llama3.2)",
        embedding_model="nomic-embed-text"
    )
    summarization_assistant = SummarizationAssistant("Ollama (llama3.2)")
    transcription_assistant = TranscriptionAssistant(model_size="base")
    
    # Set up RAG callback - this works because RAGAssistant already has setup_vectorstore_sync
    rag_config = content_processor.get_loader_config(AssistantType.RAG)
    if rag_config:
        rag_config.process_callback = rag_assistant.setup_vectorstore_sync
    
    # Set up Summarization callback - using the temporary function for now
    summary_config = content_processor.get_loader_config(AssistantType.SUMMARIZATION)
    if summary_config:
        # Use temporary function instead of a method that doesn't exist yet
        summary_config.process_callback = process_documents_temp_summarization
        
    # Set up Transcription callback - using the temporary function for now
    transcript_config = content_processor.get_loader_config(AssistantType.TRANSCRIPTION)
    if transcript_config:
        # Use temporary function instead of a method that doesn't exist yet
        transcript_config.process_callback = process_documents_temp_transcription
       # Launch Gradio interface
    logger.info("Starting the Gradio interface of the ai-workbench")
    print(f"The app will be available at: http://localhost:7960")
    demo.launch(server_port=7960, server_name="0.0.0.0", debug=True, share=False)