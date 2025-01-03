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

# Local imports
from ai_model_core.config.settings import (
    load_config, 
    get_prompt_list, 
    update_prompt_list
)
from ai_model_core.shared_utils.factory import (
    get_model,
    get_embedding_model
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
    SummarizationAssistant
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
        docs = loader.load_documents(file_paths=file_input, urls=url_input)
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
        docs = content_loader.load_documents(file_paths=files, urls=urls)
        rag_assistant.setup_vectorstore(docs)
        rag_assistant.prompt_template = prompt_info
        rag_assistant.use_history = history_flag


        logger.info("Querying RAG assistant")
        result = await rag_assistant.query(
            message, history if history_flag else None
        )
        result = await rag_assistant.query(
            message, history if history_flag else None
        )
        return result
    except Exception as e:
        logger.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred: {str(e)}"

# Wrapper function for Gradio interface summarize_assistant:
async def summarize_wrapper(loaded_docs, model_choice, method, chunk_size,
                            chunk_overlap, max_tokens, temperature, prompt_info, language,
                            verbose):
    if loaded_docs is None or len(loaded_docs) == 0:
        return "Error: No documents loaded. Please load documents before summarizing."

    try:
        
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

        # Perform summary using the method-specific prompt
        summary = await summarizer.summarize(
            loaded_docs, 
            method=method, 
            prompt_info=prompt_info,  # Use same method-specific prompt
            language=language
        )
        return summary['final_summary']
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return f"An error occurred during summarization: {str(e)}\n\nTraceback:\n{error_trace}"

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
        # Chat Tab
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        ["Ollama (LLama3.2)", "Claude Sonnet", 
                         "Claude Sonnet beta", "Deepseek v3",
                         "Mistral (large)", "Mistral (small)",
                         "Ollama (LLama3.1)", "OpenAI GPT-4o-mini"],
                        label="Choose Model",
                        value="Ollama (LLama3.1)"
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
                        additional_inputs=[
                            model_choice, temperature, max_tokens, file_input,
                            history_flag
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
        # Vision Assistant Tab
        with gr.Tab("Vision Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", label="Upload Image", image_mode="RGB"
                    )
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
                        additional_inputs=[
                            image_input, model_choice, history_flag
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear"
                    )
                    )

            vision_clear_btn = gr.Button("Clear All")
            vision_clear_btn.click(
                fn=clear_vision_chat,
                inputs=[],
                outputs=[vision_chatbot, vision_question_input, image_input]
            )

        # RAG Assistant Tab

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
                            value="Ollama (LLama3.1)"
                        )
                        embedding_choice = gr.Dropdown(
                            [
                                    "nomic-embed-text",
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
                            choices=["similarity", "mmr",
                                     "similarity_threshold"],
                            label="Select Retriever Method",
                            value="similarity"
                        )
                        num_similar_docs = gr.Slider(
                            minimum=2, maximum=10, value=3, step=1,
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
                        height=600, show_copy_button=True,
                        show_copy_all_button=True
                    )
                    rag_text_box = gr.Textbox(
                        label="RAG input",
                        placeholder="Type your question here..."
                    )
                    chat_interface = gr.ChatInterface(
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
                            max_tokens, url_input, file_input,
                            language_choice, prompt_info, history_flag,
                            retrieval_method
                        ],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                    )
                    )
                    flag_btn = gr.Button("Flag")
                    flag_options = gr.Dropdown(
                        ["High quality", "OK", "Incorrect", "Ambiguous",
                         "Inappropriate"],
                        label="Flagging Options"
                    )

            # Connect the load_button to the load_documents_wrapper function
            load_button.click(
                fn=load_documents_wrapper,
                inputs=[url_input, file_input, chunk_size, chunk_overlap],
                outputs=[load_output, gr.State()]
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
                        file_types=[".txt", ".pdf", ".docx"],
                        file_count="single"
                    )
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
                            value="Ollama (LLama3.1)"
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
                        method = gr.Dropdown(
                                ["stuff", "map_reduce", "refine"],
                                label="Summarization Strategy",
                                value="stuff"
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
                    chunk_overlap, max_tokens, temperature, prompt_info, 
                    language_choice, verbose
                ],
                outputs=summary_output
            )

    # Set up the flagging callback
    flagging_callback.setup(
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs,
        "flagged_rag_data"
    )
    flagging_callback.setup(
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs,
        "flagged_rag_data"
    )

    # Connect the flagging button to the callback
    flag_btn.click(
        lambda *args: flagging_callback.flag(args[:-1] + (args[-1],)),
        [rag_text_box, rag_chat_bot] + chat_interface.additional_inputs +
        [flag_options],
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
