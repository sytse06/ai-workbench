import os
import logging
import sys
from typing import List, Union
from PIL import Image
import traceback
import gradio as gr
from ai_model_core.config.settings import (
    load_config, get_prompt_list, update_prompt_list
)
from ai_model_core.model_helpers import (
    ChatAssistant, PromptAssistant, VisionAssistant,
    RAGAssistant, SummarizationAssistant, TranscriptionAssistant
)
from ai_model_core.model_helpers.RAG_assistant import (
    CustomHuggingFaceEmbeddings
)
from ai_model_core import get_model, get_embedding_model
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

# Initialize assistants with default models
transcription_assistant = TranscriptionAssistant("Whisper")

# Wrapper function for loading documents (RAG and summarization)
def load_documents_wrapper(url_input, file_input, chunk_size, chunk_overlap):
    try:
        loader = EnhancedContentLoader(chunk_size, chunk_overlap)
        file_paths = file_input if isinstance(file_input, list) else [file_input] if file_input else None
        docs = loader.load_and_split_document(file_paths=file_paths, urls=url_input, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return f"Successfully loaded {len(docs)} chunks of text.", docs
    except Exception as e:
        logger.error(f"Error in load_documents: {str(e)}")
        return f"An error occurred while loading documents: {str(e)}", None

def handle_transcription(file_input, url_input, transcribe_model, language, vocal_extracter, vad, precision, device, task):
    # Process the input (file or URL) to get the audio path
    audio_path = process_input(file_input, url_input)
    
    # Update the transcription assistant with new settings
    transcription_assistant.transcribe_model = get_model(transcribe_model)
    transcription_assistant.language = language
    transcription_assistant.vocal_extracter = vocal_extracter
    transcription_assistant.vad = vad
    transcription_assistant.precision = precision
    transcription_assistant.device = device
    transcription_assistant.task_type = task
    # Process the audio
    result = await transcription_assistant.process_audio(audio_path)
    # Save subtitles
    subtitle_paths = transcription_assistant.save_subtitles(result['subtitles'], audio_path)
    return result['transcription'], subtitle_paths, result['answer']

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
            
# Variables and helper functions for Gradio interface
WHISPER_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
TASK_TYPES = ["transcribe", "translate"]
SUPPORTED_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]

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
        # Transcription Tab
        with gr.Tab("Transcription"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Input audio file
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    url_input = gr.Textbox(
                        label="Video URL",
                        placeholder="Enter video URL here"
                    )
                    with gr.Row():
                    # Model selection
                    model_choice = gr.Dropdown(
                        choices=WHISPER_SIZES,
                        value="base",
                        label="Whisper Model Size",
                        info="Larger models are more accurate but slower"
                    )                    
                    # Task type selection
                    task_type = gr.Radio(
                        choices=TASK_TYPES,
                        value="transcribe",
                        label="Task Type",
                        info="'Transcribe' keeps original language, 'Translate' converts to English"
                    )
                    with gr.Row():
                    # Language selection (important for both transcribe and translate)
                    language = gr.Dropdown(
                        choices=["Auto"] + sorted(whisper.tokenizer.LANGUAGES.keys()),
                        value="Auto",
                        label="Source Language",
                        info="Specifying the source language improves speed and accuracy"
                    )                    
                    # Output format selection
                    output_format = gr.Dropdown(
                        choices=SUPPORTED_FORMATS,
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
                            choices=[x[1] for x in LANGUAGE_CODES],
                            type="index",
                            label="Language",
                            info="Select the desired audio language to improve speed."
                        )
                        vocal_extracter_checkbox = gr.Checkbox(
                            value=True,
                            label="Vocal extracter",
                            info="Mute non-vocal background music"
                        )
                        vad_checkbox = gr.Checkbox(
                            value=True,
                            label="Voice activity detection",
                            info="Should fix the issue of subtitle repetition"
                        )
                        precision_input = gr.Dropdown(
                            choices=[
                                "Low",
                                "Medium-Low",
                                "Medium",
                                "Medium-High (Recommend)",
                                "High"
                            ],
                            type="index",
                            value="Medium-High (Recommend)",
                            label="Precision",
                            info="Higher precision requires more time."
                        )
                        device_input = gr.Radio(
                            value="CPU",
                            choices=["CPU", "GPU"],
                            type="index",
                            label="Device",
                            info="GPU support requires additional setup."
                        )
                        task_type = gr.Radio(
                            choices=["Transcribe", "Translate"],
                            type="index",
                            value="Transcribe",
                            label="Task",
                            info="Translation is built-in but may be less accurate than specialized tools."
                        )
                    
                    transcribe_button = gr.Button("Start Transcription")

                with gr.Column(scale=2):
                    video_output = gr.Video(label="Transcribed Video", visible=False)
                    audio_output = gr.Audio(label="Transcribed Audio", visible=False)
                    
                    with gr.Accordion("Subtitle Downloads", open=False):
                        srt_download = gr.File(label="SRT Download")
                        vtt_download = gr.File(label="VTT Download")
                        ass_download = gr.File(label="ASS Download")
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
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True, share=False)