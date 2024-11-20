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
async def transcription_wrapper_with_progress(
    media_input,
    url_input,
    model_choice,
    language,
    task_type,
    device_input,
    output_format,
    temperature=0.0,
    verbose=True,
):
    """
    Enhanced wrapper function to handle transcription requests through Gradio interface
    with progress bar. Returns transcription results and generated subtitle files.
    """
    try:
        async def progress_callback(progress_value, status, current_text, processed_time):
            # Update Gradio components with normalized progress (0-1)
            gr.Progress().update(progress_value * 0.01)  # Convert percentage to 0-1 scale
            return {
                status_text: status,
                subtitle_preview: current_text,
                time_info: processed_time
            }

        # First, load and preprocess the audio
        loader = EnhancedContentLoader()
        audio_path = None

        try:
            if media_input:
                audio_path = media_input
            elif url_input:
                # Assume the loader can handle URL downloads
                docs = loader.preprocess_audio(urls=url_input)
                if docs:
                    audio_path = docs[0].metadata["processed_path"]

            if not audio_path:
                await progress_callback(0, "Error: No input provided", "", "")
                return (
                    "Please provide either a media file or a valid URL.",
                    None, None, None, None, None, None, None
                )
            # Create output directory if it doesn't exist
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)

            # Get the base filename without extension
            base_filename = Path(audio_path).stem
            output_base_path = output_dir / base_filename

            # Create a basic TranscriptionContext
            context = TranscriptionContext(
                speakers=[],  # Could be populated from UI in future
                terms={},     # Could be populated from UI in future
                context=""    # Could be populated from UI in future
            )

            # Initialize transcription assistant
            transcription_assistant = TranscriptionAssistant(
                model_size=model_choice.split()[-1].lower(),
                language=language,
                task_type=task_type,
                device=device_input.lower(),
                temperature=temperature,
                output_dir="./output",
                context=context,
                verbose=verbose
            )

            # Process the audio with context and progress updates
            result = await transcription_assistant.process_audio(
                audio_path,
                context=context,
                progress_callback=progress_callback
            )
            # Generate output paths and verify they exist
            def get_output_path(ext):
                path = output_base_path.with_suffix(f'.{ext}')
                return str(path) if path.exists() else None


            return (
                result["transcription"],
                audio_path if task_type == "transcribe" else None,
                None,  # video_output
                get_output_path('txt') if 'txt' in output_format else None,
                get_output_path('srt') if 'srt' in output_format else None,
                get_output_path('vtt') if 'vtt' in output_format else None,
                get_output_path('tsv') if 'tsv' in output_format else None,
                get_output_path('json') if 'json' in output_format else None,
            )

        except Exception as e:
            await progress_callback(0, f"Error: {str(e)}", "", "")
            return (str(e), *([None] * 7))

    except Exception as e:
        logger.error(f"Unexpected error in transcription wrapper: {str(e)}")
        return (f"Error: {str(e)}", *([None] * 7)) 

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# AI WorkBench")
    gr.Markdown("### Get work done with LLM's of choice")

    with gr.Tabs():
        # Transcription Tab
        with gr.Tab("Transcription"):
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
                    
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=["Auto", "en", "nl", "de", "fr", "bg"],
                            value="nl",
                            label="Source Language",
                            info="Specification improves speed"
                        )
                        output_format = gr.Dropdown(
                            choices=["txt", "srt", "vtt", 
                                     "tsv", "json", "all"],
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

                with gr.Column(scale=2):
                    # Add progress indicators
                    # Progress indicators
                    progress_bar = gr.Progress()  # Remove label parameter
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    time_info = gr.Textbox(
                        label="Processing Time",
                        interactive=False
                    )
                    # Output displays
                    video_output = gr.Video(
                        label="Transcribed Video",
                        visible=False
                    )
                    audio_output = gr.Audio(
                        label="Transcribed Audio",
                        visible=False
                    )
                    
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
            transcribe_button = gr.Button(
                "Start Transcription",
                variant="primary"
            )
            
            # Connect the transcribe button to the wrapper function
            transcribe_button.click(
                fn=transcription_wrapper_with_progress,
                inputs=[
                    media_input, url_input, model_choice, language,
                    task_type, device_input, output_format, temperature,
                    verbose
                ],
                outputs=[
                    subtitle_preview, audio_output, video_output,
                    txt_download, srt_download, vtt_download,
                    tsv_download, json_download,
                ],
                concurrency_limit=1  # Process one request at a time
            )


if __name__ == "__main__":
    logger.info("Starting the Gradio interface for transcription")
    demo.launch(
        server_port=7861,
        debug=True,
        share=False
    )
