# model_helpers/transcription_assistant.py
# Standard library imports
import logging
from typing import TypedDict, List, Annotated, Union, Optional, Dict
from pathlib import Path
import asyncio
import traceback
import os

# Third-party imports
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from dataclasses import dataclass, field

from whisper.utils import get_writer

from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import START, StateGraph, END

# Local imports
from ..config.settings import load_config
from ..shared_utils.utils import (
    EnhancedContentLoader,
    get_prompt_template,
    _format_history
)
from ..shared_utils.factory import get_model

logger = logging.getLogger(__name__)


# Custom Exceptions for TranscriptionAssistant
class TranscriptionError(Exception):
    """Raised when there's an error during the transcription process"""
    pass


class OutputError(Exception):
    """Raised when there's an error with the output format or processing"""
    pass


class AudioProcessingError(Exception):
    """Raised when there's an error processing the audio file"""
    pass


class ModelError(Exception):
    """Raised when there's an error with the transcription model"""
    pass


class FileError(Exception):
    """Raised when there's an error handling input/output files"""
    pass


class TranscriptionState(TypedDict):
    input: str
    audio_path: str
    transcription: str
    results: dict  # Store full whisper results for different output formats
    answer: str
    all_actions: Annotated[List[str], lambda x, y: x + [y]]
    initial_prompt: str


@dataclass
class TranscriptionResult:
    """Container for transcription results"""
    text: str
    segments: List[dict]
    language: str
    raw_output: dict


@dataclass
class TranscriptionContext:
    """Container for transcription context information"""
    speakers: List[str] = field(default_factory=list)
    terms: Dict[str, str] = field(default_factory=dict)  # term: description
    context: str = ""
    
    def generate_prompt(self) -> str:
        """Generate formatted initial prompt from context information"""
        prompt_parts = []
        
        # Add speakers information
        if self.speakers:
            speakers_list = ", ".join(self.speakers)
            prefix = "Speakers in the conversation: "
            speakers_str = f"{prefix}{speakers_list}"
            prompt_parts.append(speakers_str)
        
        # Add specialized terms/vocabulary
        if self.terms:
            terms_list = [
                f"- {term}: {desc}" 
                for term, desc in self.terms.items()
            ]
            terms_str = "Specialized terms:\n" + "\n".join(terms_list)
            prompt_parts.append(terms_str)
        
        # Add additional context
        if self.context:
            prompt_parts.append(self.context)
            
        return "\n\n".join(prompt_parts)


class TranscriptionAssistant:
    def __init__(
        self,
        model=None,
        model_size="large",
        language="auto",
        task_type="transcribe",
        device="cpu",
        temperature=0.0,
        output_dir="./output",
        context: Optional[TranscriptionContext] = None,
        verbose=True
    ):
        self.context = context or TranscriptionContext()
        self.model_size = model_size
        # Set up model (use provided or load new)
        try:
            if model is not None:
                self.model = model
            else:
                # Format the model choice to include "Whisper" prefix
                model_choice = f"Whisper {model_size}"
                self.model = get_model(model_choice)
            if self.model is None:
                raise ModelError("Failed to initialize the model")
        except Exception as e:
            raise ModelError(f"Model initialization failed: {str(e)}")
            
        self.language = (
            None if language in ["auto", "Auto"]
            else language
        )
        self.task_type = task_type
        self.device = device
        self.verbose = verbose
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = StateGraph(TranscriptionState)
        self.config = load_config()
        self.content_loader = EnhancedContentLoader()
        self.setup_graph()

    def setup_graph(self):
        """Initialize and setup the state graph"""
        self.graph = StateGraph(TranscriptionState)
        
        # Add nodes
        self.graph.add_node("preprocess_audio", self.preprocess_audio)
        self.graph.add_node("transcribe_audio", self.transcribe_audio)
        self.graph.add_node("save_outputs", self.save_outputs)
        self.graph.add_node("post_process", self.post_process)

        # Build graph
        self.graph.add_edge(START, "preprocess_audio")
        self.graph.add_edge("preprocess_audio", "transcribe_audio")
        self.graph.add_edge("transcribe_audio", "save_outputs")
        self.graph.add_edge("save_outputs", "post_process")
        self.graph.add_edge("post_process", END)

        self.graph_runnable = self.graph.compile()

    async def _extract_audio_from_video(self, file_path: str) -> str:
        """Extract audio from video file asynchronously"""
        try:
            video_path = Path(file_path)
            temp_audio_path = video_path.with_suffix('.wav')
            
            # Only extract if it's a video file
            if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                logger.info(f"Extracting audio from video file: {file_path}")
                await asyncio.to_thread(
                    lambda: VideoFileClip(str(video_path)).audio.write_audiofile(str(temp_audio_path))
                )
                return str(temp_audio_path)
            return file_path
        except Exception as e:
            logger.error(f"Error extracting audio from video: {str(e)}")
            raise AudioProcessingError(f"Failed to extract audio from video: {str(e)}")
    
    async def preprocess_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Preprocess audio node in workflow"""
        try:
            audio_path = state['audio_path']
            logger.info(f"Preprocessing audio file: {audio_path}")
            # Extract audio if it's a video file
            audio_path = await self._extract_audio_from_video(audio_path)
            state['audio_path'] = audio_path  # Update the audio path in state            
            # Continue with existing preprocessing                        
            audio_np = await self._preprocess_audio_file(audio_path)
            state['input'] = audio_np
            state['all_actions'].append("audio_preprocessed")
            
            # Clean up temporary audio file if it was created
            if audio_path != state['audio_path']:
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
                    
            return state
        except Exception as e:
            logger.error(f"Error in preprocess_audio: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise AudioProcessingError(f"Preprocessing failed: {str(e)}")
                                       
    async def _preprocess_audio_file(self, audio_path: str) -> np.ndarray:
        """Internal method for audio preprocessing with support for large files"""
        try:
            # Convert to Path object and verify file exists
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info("Loading audio file with pydub")
            # Load the entire audio using pydub
            audio = await asyncio.to_thread(
                lambda: AudioSegment.from_file(str(audio_file))
            )
            
            logger.info("Converting audio to mono and 16kHz")
            # Convert to mono and set sample rate
            audio = await asyncio.to_thread(
                lambda: audio.set_channels(1).set_frame_rate(16000)
            )

            # Set the maximum segment duration in milliseconds (25 minutes)
            max_segment_duration = 25 * 60 * 1000

            # If audio is longer than max_segment_duration, process in segments
            if len(audio) > max_segment_duration:
                logger.info("Processing large audio file in segments")
                
                # Convert entire audio to numpy array
                logger.info("Converting audio to numpy array")
                audio_np = (
                    np.array(audio.get_array_of_samples(), dtype=np.float32) 
                    / 32767.0
                )
                
                # Return the full numpy array for segmentation in transcribe_audio
                return audio_np
            else:
                # For shorter files, process normally
                logger.info("Processing audio as single segment")
                samples = audio.get_array_of_samples()
                return await asyncio.to_thread(
                    lambda: np.array(samples, dtype=np.float32) / 32767.0
                )

        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise AudioProcessingError(f"Audio file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error in _preprocess_audio_file: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
                                       
    async def process_audio(
        self, 
        audio_path: Union[str, Path],
        context: Optional[TranscriptionContext] = None
    ) -> dict:
        """Main entry point with context support"""
        try:
            logger.info(f"Starting audio processing for: {audio_path}")
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Use provided context or fall back to default
            current_context = context or self.context

            # Initialize state with empty input
            # Will be populated during preprocessing
            initial_state = TranscriptionState(
                input="",
                audio_path=str(audio_path),
                transcription="",
                results={},
                answer="",
                all_actions=[],
                initial_prompt=current_context.generate_prompt()
            )

            logger.info("Starting graph execution")
            final_state = await self.graph_runnable.ainvoke(initial_state)
            logger.info("Processing completed successfully")
            return self._prepare_response(final_state)

        except FileNotFoundError as e:
            logger.error(f"File error: {str(e)}")
            raise TranscriptionError(f"File error: {str(e)}")
        except AudioProcessingError as e:
            logger.error(f"Audio processing error: {str(e)}")
            raise TranscriptionError(f"Audio processing error: {str(e)}")
        except ModelError as e:
            logger.error(f"Model error: {str(e)}")
            raise TranscriptionError(f"Model error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in process_audio: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise TranscriptionError(f"Processing failed: {str(e)}")

    async def transcribe_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Transcribe audio with context support and large file handling"""
        try:
            if self.model is None:
                raise ModelError("Model not properly initialized")
                
            transcribe_func = (
                self.model.translate if self.task_type == 'translate'
                else self.model.transcribe
            )
            
            # Get the audio data from input field
            audio_np = state['input']
            
            # Set max segment duration in samples (25 minutes at 16kHz)
            max_segment_samples = 25 * 60 * 16000
            
            # If audio is longer than max_segment_samples, process in segments
            if len(audio_np) > max_segment_samples:
                logger.info("Transcribing large audio file in segments")
                
                # Initialize lists for results
                all_segments = []
                full_text = []
                
                # Process segments
                total_segments = len(audio_np) // max_segment_samples + 1
                for i, start in enumerate(
                    range(0, len(audio_np), max_segment_samples)
                ):
                    logger.info(f"Processing segment {i+1}/{total_segments}")
                    end = min(start + max_segment_samples, len(audio_np))
                    segment = audio_np[start:end]
                    
                    try:
                        # Transcribe segment
                        segment_result = await asyncio.to_thread(
                            transcribe_func,
                            segment,
                            language=self.language,
                            temperature=self.temperature,
                            initial_prompt=state.get('initial_prompt', '')
                        )
                    except Exception as e:
                        logger.error(
                            f"Error transcribing segment {i+1}: {str(e)}"
                        )
                        raise ModelError(
                            f"Failed to transcribe segment {i+1}: {str(e)}"
                        )
                    
                    # Adjust timestamps for segments after the first
                    if start > 0:
                        # Convert samples to seconds
                        segment_offset = start / 16000
                        for seg in segment_result["segments"]:
                            seg["start"] += segment_offset
                            seg["end"] += segment_offset
                    
                    # Collect results
                    all_segments.extend(segment_result["segments"])
                    full_text.append(segment_result["text"])
                
                # Combine results
                combined_results = {
                    "text": " ".join(full_text),
                    "segments": all_segments,
                    # Use language from last segment
                    "language": segment_result["language"]
                }
                
                state['transcription'] = combined_results["text"]
                state['results'] = combined_results
            else:
                # For shorter files, process normally
                logger.info("Processing audio as single segment")
                results = await asyncio.to_thread(
                    transcribe_func,
                    audio_np,
                    language=self.language,
                    temperature=self.temperature,
                    initial_prompt=state.get('initial_prompt', '')
                )
                
                state['transcription'] = results["text"]
                state['results'] = results

            state['all_actions'].append("audio_transcribed")
            return state

        except Exception as e:
            logger.error(f"Error in transcribe_audio: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise ModelError(f"Transcription failed: {str(e)}")

    async def save_outputs(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Save outputs node in workflow"""
        try:
            base_filename = Path(state['audio_path']).stem

            # Save in all supported formats
            formats = ['txt', 'srt', 'vtt', 'tsv', 'json', 'all']
            for fmt in formats:
                writer = get_writer(fmt, str(self.output_dir))
                await asyncio.to_thread(
                    writer,
                    state['results'],
                    f"{base_filename}.{fmt}"
                )

            state['all_actions'].append("outputs_saved")
            return state

        except Exception as e:
            logger.error(f"Error in save_outputs: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise OutputError(f"Failed to save outputs: {str(e)}")

    async def post_process(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Post-processing node in workflow"""
        # Implement any post-processing logic here
        state['all_actions'].append("post_processed")
        return state
        
    def _prepare_response(self, state: TranscriptionState) -> dict:
        """Prepare final response from workflow state"""
        return {
            "transcription": state['transcription'],
            "language": state['results'].get('language', ''),
            "actions": state['all_actions'],
            "output_dir": str(self.output_dir)
        }
        
    async def query(
        self,
        question: str,
        history: List[tuple] = None,
        prompt_template: str = None
    ) -> str:
        """Method to answer questions about the transcription."""
        if not history:
            history = []

        template = (
            prompt_template if prompt_template else
            "Answer the following question about the transcription: {question}"
        )
        prompt = get_prompt_template(template, self.config)

        chain = prompt | self.model | StrOutputParser()

        response = await chain.ainvoke({
            "question": question,
            "history": _format_history(history) if history else []
        })

        return (
            response if response is not None
            else "I'm sorry, I couldn't generate a response."
        )
