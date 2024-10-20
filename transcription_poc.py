# Transcription gradio experiment
# Standard library imports
import logging
import os
import sys
import asyncio
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

# Initialize the assistant with the model from get_model
def initialize_transcription_assistant(model_choice):
    try:
        # Get the model using the general get_model function
        model = get_model(model_choice)
        
        # Initialize the assistant with the loaded model
        return TranscriptionAssistant(
            model=model,
            model_size=model_choice.split()[-1].lower(),  # Keep track of size for reference
            language="auto",
            task_type="transcribe",
            output_dir="./output"
        )
    except Exception as e:
        logger.error(f"Error initializing transcription assistant: {str(e)}")
        raise

# Initialize the assistant when the app starts
transcription_assistant = initialize_transcription_assistant("Whisper base")

# Add model change handler
def handle_model_change(new_model_choice):
    global transcription_assistant
    try:
        transcription_assistant = initialize_transcription_assistant(new_model_choice)
        return f"Successfully loaded {new_model_choice}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Connect model change handler to dropdown
model_choice.change(
    fn=handle_model_change,
    inputs=[model_choice],
    outputs=[gr.Textbox(label="Model Status")]
)

# Wrapper function for handling audio files for transcription
async def load_audio_wrapper(url_input, file_input):
    """
    Wrapper function to load and preprocess audio files for transcription.
    Returns the processed audio path and any error messages.
    """
    try:
        loader = EnhancedContentLoader()
        docs = loader.preprocess_audio(file_paths=file_input, urls=url_input)
        
        if not docs:
            return None, "No audio files were successfully processed."
            
        # For now, just handle the first audio file
        processed_path = docs[0].metadata["processed_path"]
        return processed_path, f"Successfully processed audio file: {Path(file_input).name}"
        
    except Exception as e:
        logger.error(f"Error in load_audio_wrapper: {str(e)}")
        return None, f"Error processing audio: {str(e)}"

async def handle_transcription(file_input, url_input, model_choice, language, vad, device, task_type, output_format):
    try:
        # First, load and preprocess the audio
        audio_path, message = await load_audio_wrapper(url_input, file_input)
        if not audio_path:
            return None, message
            
        # Update settings
        transcription_assistant.language = language
        transcription_assistant.vad = vad
        transcription_assistant.device = device
        transcription_assistant.task_type = task_type
        
        # Process the audio
        result = await transcription_assistant.process_audio(audio_path)
        
        # Clean up temporary files
        if Path(audio_path).exists():
            Path(audio_path).unlink()
            
        return result['transcription'], result['answer']
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None, f"Error: {str(e)}"

# Second attempt to process audio
async def process_audio(
    audio_path, model_size, task_type, language, 
    output_format, temperature, vad
    ):
    try:
        if not audio_path:
            return None, None, "Please upload an audio file"
                
        # Initialize TranscriptionAssistant with selected parameters
        assistant = TranscriptionAssistant(
                    model_size=model_size,
                    language=language,
                    task_type=task_type,
                    vad=vad,
                    temperature=temperature,
                    output_dir="./output"
                )
                
        # Process the audio
        result = await assistant.process_audio(audio_path)
                
        # Get the output file path
        base_filename = Path(audio_path).stem
        output_path = f"./output/{base_filename}.{output_format}"
                
        # Generate status message
        lang_name = whisper.tokenizer.LANGUAGES.get(language, "detected language") if language == "Auto" else whisper.tokenizer.LANGUAGES.get(language)
        task_msg = "Translated to English from" if task_type == "translate" else "Transcribed"
        status_msg = f"{task_msg} {lang_name}. Output saved as {output_format}"
                
        return (
                result["transcription"],
                    output_path,
                    status_msg
                )
                
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def clear_chat():
    return None


def clear_vision_chat():
    return None, None, gr.update(value=None)

flagging_callback = gr.CSVLogger()

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
                
                with gr.Accordion("Options", open=False):
                    model_choice = gr.Dropdown(
                        choices=[f"Whisper {size}" for size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]],
                        value="Whisper large",
                        label="Whisper Model Size",
                        info="Larger models are more accurate but slower"
                    )
                    task_type = gr.Radio(
                        choices=["transcribe", "translate"],
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
                # Language helper text that updates based on selection
                language_info = gr.Markdown(visible=False)
                                    
                    with gr.Accordion("Transcription Options", open=False):
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                            info="Higher values = more random output"
                        )
                        
                        vad = gr.Checkbox(
                            value=True,
                            label="Voice Activity Detection",
                            info="Filter out non-speech segments"
                        )
                
                # Process button
                process_btn = gr.Button("Process Audio", variant="primary")
                        language_input = gr.Dropdown(
                            value="Auto",
                            choices=[x[1] for x in ["nl", "de", "fr"]],
                            type="index",
                            label="Language",
                            info="Select the audio language to improve speed."
                        )
                        task_type = gr.Radio(
                            choices=["Transcribe", "Translate"],
                            type="index",
                            value="Transcribe",
                            label="Task",
                            info="Translation is built-in but may be less accurate than specialized tools."
                        )
                        vad_checkbox = gr.Checkbox(
                            value=True,
                            label="Voice activity detection",
                            info="Should fix the issue of subtitle repetition"
                        )
                        vocal_extracter_checkbox = gr.Checkbox(
                            value=True,
                            label="Vocal extracter",
                            info="Mute non-vocal background noise"
                        )
                        device_input = gr.Radio(
                            value="CPU",
                            choices=["CPU", "GPU"],
                            type="index",
                            label="Device",
                            info="GPU support requires additional setup."
                        )                    
                    transcribe_button = gr.Button("Start Transcription")

                with gr.Column(scale=2):
                    video_output = gr.Video(label="Transcribed Video", visible=False)
                    audio_output = gr.Audio(label="Transcribed Audio", visible=False)
                    
                    with gr.Accordion("Subtitle Downloads", open=False):
                        txt_download = gr.File(label="TXT Download")
                        srt_download = gr.File(label="SRT Download")
                        vtt_download = gr.File(label="VTT Download")
                        tsv_download = gr.File(label="TSV Download")
                        json_download = gr.File(label="JSON Download")
                    
                    subtitle_preview = gr.TextArea(label="Subtitle Preview", interactive=False)

            transcribe_button.click(
                fn=handle_transcription,
                inputs=[file_input, url_input, transcribe_model_input, language_input,
                        vocal_extracter_checkbox, vad_checkbox, precision_input,
                        device_input, task_type],
                outputs=[transcription_output, subtitle_output, status_output]
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
    logger.info("Starting the Gradio interface for transcription")
    demo.launch(
        server_port=7861,
        debug=True,
        share=False
    )