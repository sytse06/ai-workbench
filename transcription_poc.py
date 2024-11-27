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
    progress=gr.Progress()
):
    try:
        # Initialize progress
        progress(0, desc="Starting transcription...")
        
        # Load and preprocess audio
        loader = EnhancedContentLoader()
        audio_path = None

        try:
            if media_input:
                audio_path = media_input
            elif url_input:
                docs = loader.preprocess_audio(urls=url_input)
                if docs:
                    audio_path = docs[0].metadata["processed_path"]

            if not audio_path:
                return (
                    "Please provide either a media file or a valid URL.",
                    None, None, None, None, None, None, None,
                    "Error: No input provided",
                    ""
                )

            # Setup output directory
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)
            base_filename = Path(audio_path).stem
            output_base_path = output_dir / base_filename
            
            context = TranscriptionContext(
                speakers=[],
                terms={},
                context=""
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

            # Create a status dictionary to store the latest updates
            status_updates = {
                "subtitle_preview": "",
                "status": "",
                "time_info": ""
            }

            async def progress_callback(percent, status, current_text, processed_time):
                nonlocal status_updates
                # Update the progress bar
                overall_progress = 0.4 + (percent * 0.5 / 100)
                progress(overall_progress, desc="Processing...")
                
                # Store the latest updates
                status_updates["status"] = status
                status_updates["time_info"] = processed_time
                if current_text:  # Only update text when we have new content
                    status_updates["subtitle_preview"] = current_text
                
                # Update Gradio components in real-time
                return [
                    status_updates["subtitle_preview"],  # subtitle_preview
                    None,  # audio_output
                    None,  # video_output
                    None,  # txt_download
                    None,  # srt_download
                    None,  # vtt_download
                    None,  # tsv_download
                    None,  # json_download
                    status_updates["status"],  # status_text
                    status_updates["time_info"],  # time_info
                ]

            result = await transcription_assistant.process_audio(
                audio_path,
                context=context,
                progress_callback=progress_callback
            )

            # Generate output paths
            def get_output_path(ext):
                path = output_base_path.with_suffix(f'.{ext}')
                return str(path) if path.exists() else None

            # Return final results
            return (
                result["transcription"],  # subtitle_preview
                audio_path if task_type == "transcribe" else None,  # audio_output
                None,  # video_output
                get_output_path('txt') if 'txt' in output_format else None,
                get_output_path('srt') if 'srt' in output_format else None,
                get_output_path('vtt') if 'vtt' in output_format else None,
                get_output_path('tsv') if 'tsv' in output_format else None,
                get_output_path('json') if 'json' in output_format else None,
                status_updates["status"],  # status_text
                status_updates["time_info"]  # time_info
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return (error_msg, None, None, None, None, None, None, None, error_msg, "")

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return (error_msg, None, None, None, None, None, None, None, error_msg, "")
            
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
                concurrency_limit=1,
                show_progress="full"  # This ensures we see all progress updates
            )


if __name__ == "__main__":
    logger.info("Starting the Gradio interface for transcription")
    demo.launch(
        server_port=7861,
        debug=True,
        share=False
    )
