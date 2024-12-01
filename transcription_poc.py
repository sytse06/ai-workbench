# Transcription gradio experiment
# Standard library imports
import logging
import os
import sys
from pathlib import Path

# Third-party imports
import gradio as gr

# Local imports
from ai_model_core.config.settings import load_config
from ai_model_core.shared_utils.utils import EnhancedContentLoader
from ai_model_core.model_helpers import TranscriptionAssistant
from ai_model_core.model_helpers.transcription_assistant import (
    TranscriptionContext,
    TranscriptionError,
    FileError,
    ModelError,
    OutputError,
    AudioProcessingError
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


# Initialize the assistant when the app starts
transcription_assistant = TranscriptionAssistant(model_size="base")

# Define empty return tuple for error cases
empty_return = (None,) * 7  # 7 None values for the 7 output fields

# Define model choices
WHISPER_SIZES = ["tiny", "base", "small", "medium"]
WHISPER_LARGE = ["large", "large-v2", "large-v3"]
ALL_SIZES = [*WHISPER_SIZES, *WHISPER_LARGE]
WHISPER_MODELS = []
for size in ALL_SIZES:
    WHISPER_MODELS.append(f"Whisper {size}")


# Wrapper function for Gradio interface transcription_assistant:
async def transcription_wrapper_streaming(
    media_input,
    url_input,
    model_choice,
    language,
    task_type,
    device_input,
    output_format,
    temperature=0.0,
    verbose=True,
    progress=gr.Progress()  # Add direct progress parameter
):
    try:
        # Initial progress
        progress(0, desc="Initializing...")
        
        transcription_assistant = TranscriptionAssistant(
            model_size=model_choice.split()[-1].lower(),
            language=language,
            task_type=task_type,
            device=device_input.lower(),
            temperature=temperature,
            output_dir="./output",
            verbose=verbose
        )
        
        # Pass progress updates through callback
        async def progress_callback(percent, status):
            progress(percent / 100, desc=status)
            
        audio_path = media_input or url_input
        if not audio_path:
            yield "Please provide input", None, None, None, None, None, None, None, "Error: No input", ""
            return

        progress(0.1, desc="Starting transcription...")
        
        # Process with streaming updates
        async for chunk_result in transcription_assistant.process_audio_streaming(
            audio_path, 
            progress_callback=progress_callback
        ):
            yield (
                chunk_result.get("current_text", ""),  # subtitle_preview
                None,                                  # audio_output
                None,                                  # video_output
                None,                                  # downloads
                None,
                None,
                None,
                None,
                chunk_result.get("status", ""),        # status_text
                chunk_result.get("processed_time", "")  # time_info
            )

    except Exception as e:
        yield str(e), None, None, None, None, None, None, None, f"Error: {str(e)}", ""
        
# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Get work done with LLM's of choice")
    with gr.Tab("Transcription"):
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
                
                with gr.Accordion("Advanced Options", open=False):
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
                        choices=["txt", "srt", "vtt", "tsv", "json", "all"],
                        value="txt",
                        label="Output Format",
                        info="Select output file format"
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
                    txt_download = gr.File(label="TXT Download")
                    srt_download = gr.File(label="SRT Download")
                    vtt_download = gr.File(label="VTT Download")
                    tsv_download = gr.File(label="TSV Download")
                    json_download = gr.File(label="JSON Download")

        # Process button under both columns
        transcribe_button = gr.Button(
            "Start Transcription",
            variant="primary"
        )            
        # Connect the transcribe button to the wrapper function
        transcribe_button.click(
            fn=transcription_wrapper_streaming,
            inputs=[
                media_input, url_input, model_choice, language,
                task_type, device_input, output_format, temperature,
                verbose
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
        )


if __name__ == "__main__":
    logger.info("Starting the Gradio interface for transcription")
    demo.launch(
        server_port=7861,
        debug=True,
        share=False
    )