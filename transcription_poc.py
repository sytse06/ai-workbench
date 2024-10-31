# Standard library imports
import logging
import os
import sys
import traceback
from typing import List, Union
from ai_model_core.model_helpers import TranscriptionAssistant
from ai_model_core.utils import EnhancedContentLoader
from pathlib import Path

# Third-party imports
import gradio as gr
from PIL import Image

# Local imports
from ai_model_core.config.settings import (
    load_config,
    get_prompt_list,
    update_prompt_list
)
from ai_model_core.factory import (  # Direct import from factory
    get_model,
    get_embedding_model
)
from ai_model_core.model_helpers import (
    ChatAssistant,
    PromptAssistant,
    RAGAssistant,
    SummarizationAssistant,
    TranscriptionAssistant,
    VisionAssistant
)
from ai_model_core.model_helpers.RAG_assistant import E5Embeddings  # Single import for E5Embeddings
from ai_model_core.utils import EnhancedContentLoader

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

# Initialize the assistant when the app starts
transcription_assistant = TranscriptionAssistant("Whisper base")

# Wrapper function for Gradio interface transcription_assistant:
async def transcription_wrapper(
    audio_input,
    url_input,
    model_choice,
    language,
    task_type,
    vad_checkbox,
    vocal_extracter_checkbox,
    device_input,
    output_format,
    temperature=0.0,
):
    """
    Wrapper function to handle transcription requests through Gradio interface.
    Returns transcription results and generated subtitle files.
    """
    try:
        # First, load and preprocess the audio
        loader = EnhancedContentLoader()
        audio_path = None
        
        if audio_input:
            audio_path = audio_input
        elif url_input:
            # Assume the loader can handle URL downloads
            docs = loader.preprocess_audio(urls=url_input)
            if docs:
                audio_path = docs[0].metadata["processed_path"]
                
        if not audio_path:
            return (
                "Please provide either an audio file or a valid URL.",
                None, None, None, None, None, None, None
            )
            
        # Create a basic TranscriptionContext
        # This can be expanded later when adding UI controls
        context = TranscriptionContext(
            speakers=[],  # Could be populated from UI in future
            terms={},     # Could be populated from UI in future
            context=""    # Could be populated from UI in future
        )
        
        # Initialize transcription assistant with selected parameters
        transcription_assistant = TranscriptionAssistant(
            model_size=model_choice.split()[-1].lower(),
            language=language,
            task_type=task_type,
            vad=vad_checkbox,
            vocal_extracter=vocal_extracter_checkbox,
            device=device_input.lower(),
            temperature=temperature,
            output_dir="./output",
            context=context  # Pass the context object
        )
        
        # Process the audio with context
        result = await transcription_assistant.process_audio(
            audio_path,
            context=context  # Pass context to process_audio
        )
        
        # Prepare file paths for outputs
        base_filename = Path(audio_path).stem
        output_dir = Path("./output")
        
        # Initialize output file paths based on format
        txt_file = str(output_dir / f"{base_filename}.txt") if output_format in ["txt", "all"] else None
        srt_file = str(output_dir / f"{base_filename}.srt") if output_format in ["srt", "all"] else None
        vtt_file = str(output_dir / f"{base_filename}.vtt") if output_format in ["vtt", "all"] else None
        tsv_file = str(output_dir / f"{base_filename}.tsv") if output_format in ["tsv", "all"] else None
        json_file = str(output_dir / f"{base_filename}.json") if output_format in ["json", "all"] else None
        
        return (
            result["transcription"],  # subtitle_preview
            audio_path if task_type == "transcribe" else None,  # audio_output
            None,  # video_output
            txt_file,  # txt_download
            srt_file,  # srt_download
            vtt_file,  # vtt_download
            tsv_file,  # tsv_download
            json_file  # json_download
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return (
            f"Error during transcription: {str(e)}",
            None, None, None, None, None, None, None
        )

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Chat with LLM's of choice and reuse prompts to get work done.")

    with gr.Tabs():
        # Transcription Tab
        with gr.Tab("Transcription"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    url_input = gr.Textbox(
                        label="Video URL",
                        placeholder="Enter video URL here"
                    )
                    
                    with gr.Row():
                        model_choice = gr.Dropdown(
                            choices=[f"Whisper {size}" for size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]],
                            value="Whisper large",
                            label="Whisper Model Size",
                            info="Larger models are more accurate but slower"
                        )
                        task_type = gr.Radio(
                            choices=["transcribe", "translate"],
                            value="transcribe",
                            label="Task Type",
                            info="'Transcribe' keeps original language, 'Translate' converts to English"
                        )
                    
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=["Auto", "nl", "de", "fr", "bg"],
                            value="Auto",
                            label="Source Language",
                            info="Specifying the source language improves speed and accuracy"
                        )
                        output_format = gr.Dropdown(
                            choices=["txt", "srt", "vtt", "tsv", "json", "all"],
                            value="txt",
                            label="Output Format",
                            info="Select output file format"
                        )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                            info="Higher values = more random output"
                        )
                        vad_checkbox = gr.Checkbox(
                            value=True,
                            label="Voice Activity Detection",
                            info="Filter out non-speech segments"
                        )
                        vocal_extracter_checkbox = gr.Checkbox(
                            value=True,
                            label="Vocal Extracter",
                            info="Mute non-vocal background noise"
                        )
                        device_input = gr.Radio(
                            choices=["CPU", "GPU"],
                            value="CPU",
                            label="Device",
                            info="GPU support requires additional setup"
                        )

                with gr.Column(scale=2):
                    # Output displays
                    video_output = gr.Video(label="Transcribed Video", visible=False)
                    audio_output = gr.Audio(label="Transcribed Audio", visible=False)
                    
                    with gr.Accordion("Downloads", open=False):
                        txt_download = gr.File(label="TXT Download")
                        srt_download = gr.File(label="SRT Download")
                        vtt_download = gr.File(label="VTT Download")
                        tsv_download = gr.File(label="TSV Download")
                        json_download = gr.File(label="JSON Download")
                    
                    subtitle_preview = gr.TextArea(
                        label="Transcription Preview",
                        interactive=False,
                        show_copy_button=True
                    )
            
            # Process button
            transcribe_button = gr.Button("Start Transcription", variant="primary")
            
            # Connect the transcribe button to the wrapper function
            transcribe_button.click(
                fn=transcription_wrapper,
                inputs=[
                    audio_input,
                    url_input,
                    model_choice,
                    language,
                    task_type,
                    vad_checkbox,
                    vocal_extracter_checkbox,
                    device_input,
                    output_format,
                    temperature,
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
                ]
            )

if __name__ == "__main__":
    logger.info("Starting the Gradio interface for transcription")
    demo.launch(server_port=7861, debug=True, share=False)