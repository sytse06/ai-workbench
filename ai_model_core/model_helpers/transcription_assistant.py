# model_helpers/transcription_assistant.py
from typing import TypedDict, List, Annotated, Union
from pathlib import Path
import asyncio
import numpy as np
from pydub import AudioSegment
import whisper
from whisper.utils import get_writer

from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

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
        max_tokens=None,
        output_dir="./output"
    ):
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

    def setup_graph(self):
        self.graph.add_node("preprocess_audio", self.preprocess_audio)
        self.graph.add_node("transcribe_audio", self.transcribe_audio)
        self.graph.add_node("save_outputs", self.save_outputs)
        self.graph.add_node("post_process", self.post_process)

        self.graph.add_edge("preprocess_audio", "transcribe_audio")
        self.graph.add_edge("transcribe_audio", "save_outputs")
        self.graph.add_edge("save_outputs", "post_process")
        self.graph.add_edge("post_process", END)

        self.graph_runnable = self.graph.compile()

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file to mono 16kHz format using pydub."""
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy array and normalize
        audio_np = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32767.0
        return audio_np

    async def transcribe_audio(self, state: TranscriptionState) -> TranscriptionState:
        """Transcribe audio using Whisper model with support for translation."""
        audio_path = state['audio_path']
        audio_np = self.preprocess_audio(audio_path)
        
        # Determine whether to transcribe or translate
        transcribe_func = self.model.translate if self.task_type == 'translate' else self.model.transcribe
        
        results = transcribe_func(
            audio_np,
            language=self.language,
            task=self.task_type,
            vad_filter=self.vad,
            temperature=self.temperature
        )
        
        return {
            "transcription": results["text"],
            "results": results,
            "all_actions": ["audio_transcribed"]
        }

    async def save_outputs(self, state: TranscriptionState) -> TranscriptionState:
        """Save transcription results in multiple formats using Whisper's get_writer."""
        base_filename = Path(state['audio_path']).stem
        
        # Save plain transcription text
        txt_path = self.output_dir / f"{base_filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(state['transcription'])

        # Save in different formats using Whisper's get_writer
        supported_formats = ['tsv', 'vtt', 'srt', 'json']
        for format_name in supported_formats:
            writer = get_writer(format_name, str(self.output_dir))
            writer(state['results'], f"{base_filename}.{format_name}")

        return {"all_actions": ["outputs_saved"]}

    async def post_process(self, state: TranscriptionState) -> TranscriptionState:
        return {
            "answer": "Transcription complete. Files saved to output directory.",
            "all_actions": ["post_processing_completed"]
        }

    async def process_audio(self, audio_path: str) -> dict:
        """Main method to process an audio file."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        initial_state = {
            "audio_path": audio_path,
            "transcription": "",
            "results": {},
            "answer": "",
            "all_actions": []
        }

        final_state = await self.graph_runnable.ainvoke(initial_state)
        return {
            "transcription": final_state["transcription"],
            "answer": final_state["answer"]
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