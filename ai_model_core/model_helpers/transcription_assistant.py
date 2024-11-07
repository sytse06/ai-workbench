# model_helpers/transcription_assistant.py
# Standard library imports
import logging
from pathlib import Path
import asyncio

# Third-party imports
import numpy as np
import gradio as gr
from pydub import AudioSegment
#import soundfile as sf
from moviepy.editor import VideoFileClip
from dataclasses import dataclass, field

import whisper
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
            speakers_str = f"Speakers in the conversation: {speakers_list}"
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
            
        # Add specialized terms/vocabulary
        if self.terms:
            terms_list = [
                f"- {term}: {desc}" 
                for term, desc in self.terms.items()
            ]
            terms_str = "Specialized terms:\n" + "\n".join(terms_list)
            prompt_parts.append(terms_str)
           
        return "\n\n".join(prompt_parts)


class TranscriptionAssistant:
    def __init__(
        self,
        model=None
        model_size="large",
        language="auto",
        task_type="transcribe",
        vocal_extracter=True,
        vad=True,
        device="cpu",
        temperature=0.0,
        output_dir="./output",
        context: Optional[TranscriptionContext] = None,
        verbose=True,
        progress: Optional[gr.Progress] = None
    ):
        self.context = context or TranscriptionContext()
        self.model_size = model_size
        self.model = (
            model if model is not None
            else whisper.load_model(model_size)
        )
        self.language = "auto" if language == "Auto" else language
        self.task_type = task_type
        self.vocal_extracter = vocal_extracter
        self.vad = vad
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

    async def preprocess_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Preprocess audio node in workflow"""
        try:
            if self.progress:
                self.progress(0.1, desc="Preprocessing audio...")
                
            audio_path = state['audio_path']
            if self.vocal_extracter:
                audio_path = await self._extract_vocals(audio_path)

            audio_np = await self._preprocess_audio_file(audio_path)
            state['input'] = audio_np  # Store the audio data in the 'input' field
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
            # Convert to Path object and verify file exists
            audio_file = Path(audio_path)
            if not audio_file.exists():
                msg = f"Audio file not found at path: {audio_path}"
                raise FileNotFoundError(msg)
            
            # Open and process the audio file
            audio = await asyncio.to_thread(
                lambda: AudioSegment.from_file(str(audio_file))
            )
            
            # Convert to mono and set sample rate
            audio = await asyncio.to_thread(
                lambda: audio.set_channels(1).set_frame_rate(16000)
            )
            
            # Convert to numpy array
            samples = audio.get_array_of_samples()
            return await asyncio.to_thread(
                lambda: np.array(samples, dtype=np.float32) / 32767.0
            )
        except FileNotFoundError as e:
            raise AudioProcessingError(f"Audio file not found: {str(e)}")
        except Exception as e:
            await self._log_progress(f"Error in preprocessing: {str(e)}", state, 0)
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
                                       
    async def _extract_vocals(self, audio_path: str) -> str:
        """Extract vocals from audio if enabled"""
        try:
            # Implement vocal extraction logic here
            # Return path to processed audio
            return audio_path  # Temporary return original path
        except Exception as e:
            raise AudioProcessingError(f"Vocal extraction failed: {str(e)}")
    
    async def process_audio(
        self, 
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable] = None
    ) -> AsyncGenerator[dict, None]:
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                msg = f"Audio file not found: {audio_path}"
                raise FileNotFoundError(msg)

            audio = AudioSegment.from_file(str(audio_path))
            total_duration = len(audio) / 1000.0
            processed_duration = 0
            full_text = []
            chunk_count = 0
            total_chunks = int(total_duration / 30) + 1

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

            timeout = self.config.get('timeout', 300)
            async with asyncio.timeout(timeout):
                final_state = await self.graph_runnable.ainvoke(initial_state)

            async for chunk in self._preprocess_audio_file(str(audio_path)):
                chunk_count += 1
                chunk_start_time = time.time()
                chunk_duration = len(chunk) / 16000
                
                try:
                    logger.info(f"Processing chunk {chunk_count}/{total_chunks}")
                    
                    # Remove initial_prompt from transcribe call
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
                    
                    chunk_time = time.time() - chunk_start_time
                    speed_ratio = chunk_duration/chunk_time
                    progress_percent = min(95, int((processed_duration / total_duration) * 100))
                    
                    if progress_callback:
                        await progress_callback(
                            progress_percent,
                            f"Processing chunk {chunk_count}/{total_chunks}"
                        )
                    
                    full_text.append(result["text"])
                    raw_text = " ".join(full_text)

                    # Format display text with context headers
                    display_text = raw_text
                    if self.context and (self.context.speakers or self.context.terms or self.context.context):
                        display_text = "\n\n".join([
                            "==============TRANSCRIPTION CONTEXT==============",
                            initial_prompt,
                            "==============TRANSCRIPTION RESULT==============",
                            raw_text
                        ])
                    
                    yield {
                        "status": f"Processing chunk {chunk_count} of {total_chunks}",
                        "current_text": display_text,  # Formatted with context
                        "raw_text": raw_text,         # Clean text for file output
                        "processed_time": f"{minutes_processed}:{seconds_processed:02d} / {minutes_total}:{seconds_total:02d}",
                        "verbose_details": f"Chunk {chunk_count}/{total_chunks} processed at {speed_ratio:.2f}x real-time speed"
                    }

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {str(e)}")
                    yield {
                        "status": f"Error in chunk {chunk_count}/{total_chunks}: {str(e)}",
                        "current_text": " ".join(full_text),
                        "raw_text": " ".join(full_text),
                        "processed_time": "",
                        "verbose_details": f"Error details: {traceback.format_exc()}"
                    }
                    raise

            logger.info("Transcription completed successfully")
            if progress_callback:
                await progress_callback(100, "Transcription complete!")

            yield {
                "status": "Transcription complete!",
                "current_text": display_text,  # Final formatted text with context
                "raw_text": raw_text,         # Final clean text for files
                "processed_time": f"{minutes_total}:{seconds_total:02d} / {minutes_total}:{seconds_total:02d}",
                "verbose_details": f"Processed {total_chunks} chunks in {minutes_total}:{seconds_total:02d}"
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            if progress_callback:
                await progress_callback(0, f"Error: {str(e)}")
            yield {
                "status": f"Error: {str(e)}",
                "current_text": "",
                "raw_text": "",
                "processed_time": "",
                "verbose_details": f"Error details: {traceback.format_exc()}"
            }
                                
    async def _log_progress(
        self, 
        message: str, 
        state: TranscriptionState, 
        progress_value: float = None
    ):
        """Enhanced progress logging with streaming support"""
        if self.verbose:
            self.logger.info(message)
            
        if state and 'progress_callback' in state:
            progress_info = {
                "status": message,
                "current_text": state.get('transcription', ''),
                "processed_time": state.get('processed_time', ''),
                "progress": progress_value if progress_value is not None else 0
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

    async def transcribe_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Transcribe audio with context support"""
        try:
            transcribe_func = (
                self.model.translate if self.task_type == 'translate'
                else self.model.transcribe
            )
            
            # Use preprocessed audio data from input field
            results = await asyncio.to_thread(
                transcribe_func,
                state['input'],
                language=self.language,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                vad_filter=self.vad,
                initial_prompt=state.get('initial_prompt', '')
            )

            await self._log_progress("Starting transcription...", state, TranscriptionStages.TRANSCRIBE[0])

            async for chunk in audio_chunks_generator:
                chunk_count += 1
                chunk_start_time = time.time()
                chunk_duration = len(chunk) / 16000

                progress = TranscriptionStages.scale_progress(
                    (processed_duration / total_duration) * 100,
                    (TranscriptionStages.TRANSCRIBE[0], 95)
                )

                await self._log_progress(f"Processing chunk {chunk_count}...", state, progress)

                try:
                    # Remove initial_prompt from transcribe call
                    chunk_result = await asyncio.to_thread(
                        transcribe_func,
                        chunk,
                        language=self.language,
                        temperature=self.temperature
                    )
                except Exception as e:
                    await self._log_progress(f"Error in chunk {chunk_count}: {str(e)}", state, progress)
                    raise ModelError(f"Failed to transcribe chunk {chunk_count}: {str(e)}")

                processed_duration += chunk_duration

                if processed_duration > chunk_duration:
                    for seg in chunk_result["segments"]:
                        seg["start"] += (processed_duration - chunk_duration)
                        seg["end"] += (processed_duration - chunk_duration)

                all_segments.extend(chunk_result["segments"])
                full_text.append(chunk_result["text"])

                # Store raw transcription for file output
                state['transcription'] = " ".join(full_text)

                # Store display version with context headers
                if self.context and (self.context.speakers or self.context.terms or self.context.context):
                    state['display_text'] = "\n\n".join([
                        "==============TRANSCRIPTION CONTEXT==============",
                        initial_prompt,
                        "==============TRANSCRIPTION RESULT==============",
                        state['transcription']
                    ])
                else:
                    state['display_text'] = state['transcription']

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

            # Update final state
            state.update({
                'results': {
                    "text": state['transcription'],  # Use clean text for file outputs
                    "segments": all_segments,
                    "language": chunk_result["language"]
                },
                'all_actions': state['all_actions'] + ["audio_transcribed"]
            })

            await self._log_progress("Transcription complete!", state, 100)
            return state

        except Exception as e:
            await self._log_progress(f"Transcription failed: {str(e)}", state)
            raise ModelError(f"Transcription failed: {str(e)}")

    async def save_outputs(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Save outputs node in workflow"""
        try:
            selected_format = state.get('selected_format', 'none')
            if selected_format == 'none':
                state['all_actions'].append("no_output_files_requested")
                await self._log_progress("No output files requested", state, 100)
                return state
                
            await self._log_progress("Starting to save outputs...", state, 90)
            
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

            await self._log_progress("All files saved", state, 100)
            state['all_actions'].append("outputs_saved")
            return state

        except Exception as e:
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
