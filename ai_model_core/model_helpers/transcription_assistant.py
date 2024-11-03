# model_helpers/transcription_assistant.py
from typing import TypedDict, List, Annotated, Union, Optional, Dict
from pathlib import Path
import asyncio
import numpy as np
from pydub import AudioSegment
from dataclasses import dataclass, field
import whisper
from whisper.utils import get_writer

from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END

from ai_model_core import get_prompt_template, _format_history
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader

class TranscriptionState(TypedDict):
    input: str
    audio_path: str
    transcription: str
    results: dict  # Store full whisper results for different output formats
    answer: str
    all_actions: Annotated[List[str], lambda x, y: x + [y]]
    initial_prompt: str    
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
            speakers_str = "Speakers in the conversation: " + ", ".join(self.speakers)
            prompt_parts.append(speakers_str)
        
        # Add specialized terms/vocabulary
        if self.terms:
            terms_str = "Specialized terms:\n" + "\n".join(
                f"- {term}: {description}" 
                for term, description in self.terms.items()
            )
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
        vocal_extracter=True,
        vad=True,
        device="cpu",
        temperature=0.0,
        max_tokens=None,
        output_dir="./output",
        context: Optional[TranscriptionContext] = None
    ):
        self.context = context or TranscriptionContext()
        self.model_size = model_size
        self.model = model if model is not None else whisper.load_model(model_size)
        self.language = "auto" if language == "Auto" else language
        self.task_type = task_type
        self.vocal_extracter = vocal_extracter
        self.vad = vad
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = StateGraph(TranscriptionState)
        self.config = load_config()
        self.content_loader = EnhancedContentLoader()
        self.setup_graph()

    def setup_graph(self):
        # Initialize state graph
        self.graph = StateGraph(TranscriptionState)
        
        #Add nodes
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

    async def preprocess_audio(self, state: TranscriptionState) -> TranscriptionState:
        """Preprocess audio node in workflow"""
        try:
            # Handle vocal extraction if enabled
            audio_path = state['audio_path']
            if self.vocal_extracter:
                audio_path = await self._extract_vocals(audio_path)

            audio_np = await self._preprocess_audio_file(audio_path)
            state['audio_data'] = audio_np
            state['all_actions'].append("audio_preprocessed")
            return state
        except Exception as e:
            raise AudioProcessingError(f"Preprocessing failed: {str(e)}")
                                       
    async def _preprocess_audio_file(self, audio_path: str) -> np.ndarray:
        """Internal method for audio preprocessing"""
        try:
            audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
            audio = await asyncio.to_thread(
                lambda: audio.set_channels(1).set_frame_rate(16000)
            )
            return await asyncio.to_thread(
                lambda: np.array(audio.get_array_of_samples(), dtype=np.float32) / 32767.0
            )
        except Exception as e:
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
                                       
    async def _extract_vocals(self, audio_path: str) -> str:
        """Extract vocals from audio if enabled"""
        try:
            # Implement vocal extraction logic here
            # Return path to processed audio
            pass
        except Exception as e:
            raise AudioProcessingError(f"Vocal extraction failed: {str(e)}")
    
    async def process_audio(
        self, 
        audio_path: Union[str, Path],
        context: Optional[TranscriptionContext] = None
    ) -> dict:
        """Main entry point with context support"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Use provided context or fall back to default
            current_context = context or self.context

            initial_state = TranscriptionState(
                audio_path=str(audio_path),
                audio_data=None,
                transcription="",
                results={},
                all_actions=[],
                initial_prompt=current_context.generate_prompt()
            )

            async with asyncio.timeout(self.config.get('timeout', 300)):
                final_state = await self.graph_runnable.ainvoke(initial_state)

            return self._prepare_response(final_state)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise TranscriptionError(f"Processing failed: {str(e)}")

    async def transcribe_audio(self, state: TranscriptionState) -> TranscriptionState:
        """Transcribe audio with context support"""
        try:
            transcribe_func = self.model.translate if self.task_type == 'translate' else self.model.transcribe
            
            results = await asyncio.to_thread(
                transcribe_func,
                state['audio_data'],
                language=self.language,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                vad_filter=self.vad,
                initial_prompt=state.get('initial_prompt', '')  # Add initial prompt
            )

            state['transcription'] = results["text"]
            state['results'] = results
            state['all_actions'].append("audio_transcribed")
            return state

        except Exception as e:
            raise ModelError(f"Transcription failed: {str(e)}")

    async def save_outputs(self, state: TranscriptionState) -> TranscriptionState:
        """Save outputs node in workflow"""
        try:
            base_filename = Path(state['audio_path']).stem
            result = TranscriptionResult(
                text=state['transcription'],
                segments=state['results'].get('segments', []),
                language=state['results'].get('language', ''),
                raw_output=state['results']
            )

            # Save in all supported formats using Whisper's writers directly
            formats = ['txt', 'srt', 'vtt', 'tsv', 'json', 'all']
            for fmt in formats:
                writer = get_writer(fmt, str(self.output_dir))
                await asyncio.to_thread(
                    writer,
                    state['results'],  # Use raw results directly
                    f"{base_filename}.{fmt}"
                )

            state['all_actions'].append("outputs_saved")
            return state

        except Exception as e:
            raise OutputError(f"Failed to save outputs: {str(e)}")

    async def post_process(self, state: TranscriptionState) -> TranscriptionState:
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
        
    async def query(self, question: str, history: List[tuple] = None, prompt_template: str = None) -> str:
        """Method to answer questions about the transcription."""
        if not history:
            history = []

        prompt = get_prompt_template(prompt_template, self.config) if prompt_template else ChatPromptTemplate.from_template(
            "Answer the following question about the transcription: {question}"
        )

        chain = (
            prompt
            | self.model
            | StrOutputParser()
        )

        response = await chain.ainvoke({
            "question": question,
            "history": _format_history(history) if history else []
        })

        return response if response is not None else "I'm sorry, I couldn't generate a response. Please try rephrasing your question."