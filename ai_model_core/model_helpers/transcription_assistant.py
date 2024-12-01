# model_helpers/transcription_assistant.py
# Standard library imports
import logging
from pathlib import Path
import asyncio
import traceback
import os
from typing import (
   TypedDict, 
   List, 
   Annotated, 
   Union, 
   Optional, 
   Dict,
   Callable, 
   AsyncGenerator
)
import time

# Third-party imports
import numpy as np
import gradio as gr
from pydub import AudioSegment
#import soundfile as sf
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

class TranscriptionStages:
    PREPROCESS_FILE = (0, 10)    # 0-10%
    PREPROCESS_AUDIO = (10, 15)  # 10-15%
    TRANSCRIBE = (15, 100)       # 15-100%
    
    @staticmethod
    def scale_progress(progress: float, stage: tuple) -> float:
        """Scale progress within a stage to overall progress"""
        start, end = stage
        return start + (progress * (end - start) / 100)

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
        verbose=True,
        progress: Optional[gr.Progress] = None
    ):
        self.context = context or TranscriptionContext()
        self.model_size = model_size
        self.progress = progress
        # Set up model (use provided or load new)
        try:
            if model is not None:
                self.model = model
            else:
                # Use get_model with full model choice string
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
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)        
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

    async def _stream_extract_audio_from_video(self, file_path: str) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio extraction from video file"""
        try:
            video_path = Path(file_path)
            
            if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                logger.info(f"Streaming audio from video file: {file_path}")
                
                # Open video file
                with VideoFileClip(str(video_path)) as video:
                    if video.audio is None:
                        raise AudioProcessingError("No audio stream found in video file")
                    
                    # Define chunk size
                    chunk_duration = 30  # 30 seconds
                    
                    # Process audio in chunks
                    for t in range(0, int(video.duration), chunk_duration):
                        end_t = min(t + chunk_duration, video.duration)
                        
                        # Extract audio chunk
                        chunk = video.audio.subclip(t, end_t)
                        
                        # Convert to numpy array
                        chunk_array = np.array(chunk.to_soundarray(fps=16000))
                        
                        # Ensure mono
                        if len(chunk_array.shape) > 1:
                            chunk_array = chunk_array.mean(axis=1)
                        
                        yield chunk_array
                        
        except Exception as e:
            logger.error(f"Error streaming audio from video: {str(e)}")
            raise AudioProcessingError(f"Failed to stream audio from video: {str(e)}")

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
        
    async def preprocess_audio(self, state: TranscriptionState) -> TranscriptionState:
        try:
            if self.progress:
                self.progress(0.1, desc="Preprocessing audio...")
                
            audio_path = state['audio_path']
            logger.info(f"Preprocessing audio file: {audio_path}")
            
            audio_path = await self._extract_audio_from_video(audio_path)
            state['audio_path'] = audio_path
            
            if self.progress:
                self.progress(0.2, desc="Audio extraction complete")
            
            audio_chunks = []
            async for chunk in self._preprocess_audio_file(audio_path):
                audio_chunks.append(chunk)
            
            state['input'] = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            state['all_actions'].append("audio_preprocessed")
            
            if audio_path != state['audio_path']:
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
            
            if self.progress:
                self.progress(0.3, desc="Preprocessing complete")
                
            return state
            
        except Exception as e:
            logger.error(f"Error in preprocess_audio: {str(e)}")
            raise AudioProcessingError(f"Preprocessing failed: {str(e)}")
                                                   
    async def _preprocess_audio_file(
        self, 
        audio_path: str, 
        state: Optional[TranscriptionState] = None
    ) -> AsyncGenerator[np.ndarray, None]:
        # Scale progress within PREPROCESS_FILE stage (0-10%)
        try:
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            chunk_duration = 30 * 1000  # 30s chunks
            
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_channels(1).set_frame_rate(16000)
            total_duration = len(audio)
            
            try:
                for start_ms in range(0, total_duration, chunk_duration):
                    chunk_number = start_ms // chunk_duration
                    chunk_progress = (start_ms / total_duration) * 20
                    
                    await self._log_progress(
                        f"Preprocessing chunk {chunk_number + 1}...",
                        state,
                        TranscriptionStages.scale_progress(chunk_progress, TranscriptionStages.PREPROCESS_FILE)
                    )
                    
                    chunk = audio[start_ms:min(start_ms + chunk_duration, total_duration)]
                    chunk_samples = np.array(chunk.get_array_of_samples(), dtype=np.float32) / 32767.0
                    yield chunk_samples
                    
            finally:
                del audio
                
        except Exception as e:
            await self._log_progress(f"Error in preprocessing: {str(e)}", state, 0)
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
            
    async def process_audio(
        self, 
        audio_path: Union[str, Path],
        context: Optional[TranscriptionContext] = None,
        progress_callback: Optional[Callable] = None
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
            initial_state = TranscriptionState(
                input="",
                audio_path=str(audio_path),
                transcription="",
                results={},
                answer="",
                all_actions=[],
                initial_prompt=current_context.generate_prompt()
            )

            # Pass progress_callback to transcribe_audio through state
            if progress_callback:
                initial_state['progress_callback'] = progress_callback

            logger.info("Starting graph execution")
            final_state = await self.graph_runnable.ainvoke(initial_state)
            logger.info("Processing completed successfully")
            return self._prepare_response(final_state)

        except Exception as e:
            logger.error(f"Unexpected error in process_audio: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise TranscriptionError(f"Processing failed: {str(e)}")

    async def process_audio_streaming(
        self, 
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable] = None
    ) -> AsyncGenerator[dict, None]:
        """Streaming processing with differentiated progress updates"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.error(f"File not found: {audio_path}")
                yield {
                    "status": "Error: File not found",
                    "current_text": "",
                    "processed_time": ""
                }
                return

            # Get audio duration for progress tracking
            audio = AudioSegment.from_file(str(audio_path))
            total_duration = len(audio) / 1000.0
            processed_duration = 0
            full_text = []
            chunk_count = 0
            total_chunks = int(total_duration / 30) + 1  # Assuming 30s chunks

            logger.info(f"Starting transcription of {audio_path.name}")
            if progress_callback:
                await progress_callback(0, f"Preparing to process {total_chunks} chunks...")

            yield {
                "status": "Initializing transcription...",
                "current_text": "",
                "processed_time": f"0:00 / {int(total_duration//60)}:{int(total_duration%60):02d}"
            }

            # Process audio in chunks
            async for chunk in self._preprocess_audio_file(str(audio_path)):
                chunk_count += 1
                chunk_duration = len(chunk) / 16000
                
                try:
                    logger.info(f"Processing chunk {chunk_count}/{total_chunks}")
                    
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        chunk,
                        language=self.language,
                        temperature=self.temperature
                    )
                    
                    processed_duration += chunk_duration
                    minutes_processed = int(processed_duration // 60)
                    seconds_processed = int(processed_duration % 60)
                    minutes_total = int(total_duration // 60)
                    seconds_total = int(total_duration % 60)
                    
                    progress_percent = min(95, int((processed_duration / total_duration) * 100))
                    
                    if progress_callback:
                        await progress_callback(
                            progress_percent, 
                            f"Processing chunk {chunk_count}/{total_chunks}"
                        )
                    
                    full_text.append(result["text"])
                    
                    # Differentiated updates for different UI components
                    yield {
                        "status": f"Processing chunk {chunk_count} of {total_chunks}",
                        "current_text": " ".join(full_text),
                        "processed_time": (
                            f"{minutes_processed}:{seconds_processed:02d} / "
                            f"{minutes_total}:{seconds_total:02d}"
                        )
                    }

                    logger.info(
                        f"Chunk {chunk_count}/{total_chunks} completed - "
                        f"Time: {minutes_processed}:{seconds_processed:02d} / "
                        f"{minutes_total}:{seconds_total:02d}"
                    )

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {str(e)}")
                    if progress_callback:
                        await progress_callback(0, f"Error in chunk {chunk_count}")
                    yield {
                        "status": f"Error in chunk {chunk_count}/{total_chunks}: {str(e)}",
                        "current_text": " ".join(full_text),
                        "processed_time": ""
                    }
                    raise

            # Final status
            logger.info("Transcription completed successfully")
            if progress_callback:
                await progress_callback(100, "Transcription complete!")

            yield {
                "status": "Transcription complete!",
                "current_text": " ".join(full_text),
                "processed_time": f"{minutes_total}:{seconds_total:02d} / {minutes_total}:{seconds_total:02d}"
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            if progress_callback:
                await progress_callback(0, f"Error: {str(e)}")
            yield {
                "status": f"Error: {str(e)}",
                "current_text": "",
                "processed_time": ""
            }
                        
    async def _log_progress(self, message: str, state: TranscriptionState, progress_value: float):
        """Enhanced progress logging with streaming support"""
        if self.verbose:
            self.logger.info(message)
            
        if state and 'progress_callback' in state:
            progress_info = {
                "status": message,
                "current_text": state.get('transcription', ''),
                "processed_time": state.get('processed_time', ''),
                "progress": progress_value
            }
            try:
                await state['progress_callback'](progress_info)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {str(e)}")
                        
    async def update_progress(
        self,
        state: TranscriptionState,
        chunk_count: int,
        processed_duration: float,
        total_duration: float,
        status: str = None
    ) -> dict:
        """Enhanced progress update with detailed information"""
        minutes_processed = int(processed_duration // 60)
        seconds_processed = int(processed_duration % 60)
        minutes_total = int(total_duration // 60)
        seconds_total = int(total_duration % 60)
        
        progress_info = {
            'percent': min(100, int((processed_duration / total_duration) * 100)),
            'status': status or f"Processing chunk {chunk_count}",
            'current_text': state.get('transcription', ''),
            'processed_time': (
                f"{minutes_processed}:{seconds_processed:02d} / "
                f"{minutes_total}:{seconds_total:02d}"
            )
        }
        
        if 'progress_callback' in state:
            try:
                await state['progress_callback'](
                    progress_info['percent'],
                    progress_info['status'],
                    progress_info['current_text'],
                    progress_info['processed_time']
                )
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {str(e)}")
        
        return progress_info

    async def transcribe_audio(self, state: TranscriptionState) -> TranscriptionState:
        try:
            if self.model is None:
                raise ModelError("Model not properly initialized")

            transcribe_func = (
                self.model.translate if self.task_type == 'translate'
                else self.model.transcribe
            )

            audio = AudioSegment.from_file(state['audio_path'])
            total_duration = len(audio) / 1000.0
            processed_duration = 0
            
            audio_chunks_generator = self._preprocess_audio_file(state['audio_path'], state)
            all_segments = []
            full_text = []
            chunk_count = 0

            # Initial progress at start of transcription stage
            await self._log_progress(
                "Starting transcription...",
                state,
                TranscriptionStages.TRANSCRIBE[0]  # 15%
            )

            async for chunk in audio_chunks_generator:
                chunk_count += 1
                chunk_start_time = time.time()
                chunk_duration = len(chunk) / 16000

                # Scale progress between 15-95%
                progress = TranscriptionStages.scale_progress(
                    (processed_duration / total_duration) * 100,
                    (TranscriptionStages.TRANSCRIBE[0], 95)
                )

                await self._log_progress(
                    f"Processing chunk {chunk_count}...",
                    state,
                    progress
                )

                try:
                    chunk_result = await asyncio.to_thread(
                        transcribe_func,
                        chunk,
                        language=self.language,
                        temperature=self.temperature,
                        initial_prompt=state.get('initial_prompt', '')
                    )
                except Exception as e:
                    await self._log_progress(f"Error in chunk {chunk_count}: {str(e)}", state, progress)
                    raise ModelError(f"Failed to transcribe chunk {chunk_count}: {str(e)}")

                processed_duration += chunk_duration

                # Adjust timestamps for continuous audio
                if processed_duration > chunk_duration:
                    for seg in chunk_result["segments"]:
                        seg["start"] += (processed_duration - chunk_duration)
                        seg["end"] += (processed_duration - chunk_duration)

                all_segments.extend(chunk_result["segments"])
                full_text.append(chunk_result["text"])
                state['transcription'] = " ".join(full_text)

                chunk_time = time.time() - chunk_start_time
                speed_ratio = chunk_duration/chunk_time

                await self._log_progress(
                    f"Chunk {chunk_count} processed at {speed_ratio:.2f}x real-time speed",
                    state,
                    progress
                )

                state['processed_time'] = f"{int(processed_duration)}/{int(total_duration)} seconds"

            if not full_text:
                raise ModelError("No chunks were processed successfully")

            state.update({
                'transcription': " ".join(full_text),
                'results': {
                    "text": " ".join(full_text),
                    "segments": all_segments,
                    "language": chunk_result["language"]
                },
                'all_actions': state['all_actions'] + ["audio_transcribed"]
            })

            await self._log_progress("Transcription complete!", state, 100)
            return state

        except Exception as e:
            await self._log_progress(f"Transcription failed: {str(e)}", state, TranscriptionStages.TRANSCRIBE[0])
            raise ModelError(f"Transcription failed: {str(e)}")
        
    async def _cleanup_resources(self, state: TranscriptionState):
        """Clean up any temporary files or resources"""
        try:
            if 'temp_files' in state:
                for temp_file in state['temp_files']:
                    try:
                        os.remove(temp_file)
                        logger.info(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up file {temp_file}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    async def save_outputs(self, state: TranscriptionState) -> TranscriptionState:
        """Enhanced output saving with progress updates"""
        try:
            await self._log_progress("Saving outputs...", state, 90)
            
            base_filename = Path(state['audio_path']).stem

            formats = ['txt', 'srt', 'vtt', 'tsv', 'json', 'all']
            for i, fmt in enumerate(formats):
                progress = 90 + ((i + 1) / len(formats)) * 10
                await self._log_progress(f"Saving {fmt} format...", state, progress)
                
                writer = get_writer(fmt, str(self.output_dir))
                await asyncio.to_thread(writer, state['results'], f"{base_filename}.{fmt}")

            state['all_actions'].append("outputs_saved")
            await self._log_progress("All outputs saved successfully!", state, 100)
            
            return state

        except Exception as e:
            await self._log_progress(f"Error saving outputs: {str(e)}", state)
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
