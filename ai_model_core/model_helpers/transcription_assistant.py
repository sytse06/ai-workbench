# model_helpers/transcription_assistant.py
# Standard library imports
import logging
from typing import TypedDict, List, Annotated, Union, Optional, Dict
from pathlib import Path
import asyncio

# Third-party imports
import numpy as np
from pydub import AudioSegment
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

    async def preprocess_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Preprocess audio node in workflow"""
        try:
            audio_np = await self._preprocess_audio_file(state['audio_path'])
            state['input'] = audio_np  # Store audio data in input field
            state['all_actions'].append("audio_preprocessed")
            return state
        except Exception as e:
            raise AudioProcessingError(f"Preprocessing failed: {str(e)}")
                                       
    async def _preprocess_audio_file(self, audio_path: str) -> np.ndarray:
        """Internal method for audio preprocessing"""
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
            msg = f"Failed to preprocess audio: {str(e)}"
            raise AudioProcessingError(msg)
                                       
    async def process_audio(
        self, 
        audio_path: Union[str, Path],
        context: Optional[TranscriptionContext] = None
    ) -> dict:
        """Main entry point with context support"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                msg = f"Audio file not found: {audio_path}"
                raise FileNotFoundError(msg)

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

            timeout = self.config.get('timeout', 300)
            async with asyncio.timeout(timeout):
                final_state = await self.graph_runnable.ainvoke(initial_state)

            return self._prepare_response(final_state)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise TranscriptionError(f"Processing failed: {str(e)}")

    async def transcribe_audio(
        self,
        state: TranscriptionState
    ) -> TranscriptionState:
        """Transcribe audio with context support"""
        try:
            if self.model is None:
                raise ModelError("Model not properly initialized")
                
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
                initial_prompt=state.get('initial_prompt', '')
            )

            state['transcription'] = results["text"]
            state['results'] = results
            state['all_actions'].append("audio_transcribed")
            return state

        except Exception as e:
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
