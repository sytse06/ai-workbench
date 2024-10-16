# model_helpers/transcription_assistant.py
from typing import TypedDict, List, Annotated, Union
from pathlib import Path
import asyncio

from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ai_model_core import get_model, get_prompt_template, _format_history
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader

import whisper
import pysubs2

class TranscriptionState(TypedDict):
    input: str
    audio_path: str
    transcription: str
    subtitles: dict
    answer: str
    all_actions: Annotated[List[str], lambda x, y: x + [y]]

class TranscriptionAssistant:
    def __init__(
        self,
        transcribe_model="Whisper base",
        language="auto",
        task_type="transcribe",
        vocal_extracter=True,
        vad=True,
        precision="medium",
        device="cpu",
        temperature=0.0,
        max_tokens=None
    ):
        self.transcribe_model = get_model(transcribe_model)
        self.language = "auto" if language == "Auto" else language
        self.task_type = task_type
        self.vocal_extracter = vocal_extracter
        self.vad = vad
        self.precision = precision
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.graph = StateGraph(TranscriptionState)
        self.config = load_config()
        self.content_loader = EnhancedContentLoader()

    def setup_graph(self):
        self.graph.add_node("transcribe_audio", self.transcribe_audio)
        self.graph.add_node("generate_subtitles", self.generate_subtitles)
        self.graph.add_node("post_process", self.post_process)

        self.graph.add_edge("transcribe_audio", "generate_subtitles")
        self.graph.add_edge("generate_subtitles", "post_process")
        self.graph.add_edge("post_process", END)

        self.graph_runnable = self.graph.compile()

    async def transcribe_audio(self, state: TranscriptionState) -> TranscriptionState:
        audio_path = state['audio_path']
        result = self.transcribe_model.transcribe(
            audio_path,
            language=self.language,
            task=self.task_type,
            vad_filter=self.vad,
            temperature=self.temperature
        )
        return {"transcription": result["text"], "all_actions": ["audio_transcribed"]}

    async def generate_subtitles(self, state: TranscriptionState) -> TranscriptionState:
        transcription = state['transcription']
        segments = self.transcribe_model.transcribe(state['audio_path'])['segments']
        
        subs = pysubs2.SSAFile()
        for segment in segments:
            start = int(segment['start'] * 1000)
            end = int(segment['end'] * 1000)
            text = segment['text'].strip()
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))

        subtitles = {
            'srt': subs.to_string('srt'),
            'vtt': subs.to_string('vtt'),
            'ass': subs.to_string('ass'),
            'json': subs.to_string('json')
        }
        
        return {"subtitles": subtitles, "all_actions": ["subtitles_generated"]}

    async def post_process(self, state: TranscriptionState) -> TranscriptionState:
        # Here you can add any post-processing steps, such as cleaning up the transcription
        # or applying any additional formatting
        return {"answer": "Transcription and subtitles generation completed.", "all_actions": ["post_processing_completed"]}

    async def process_audio(self, audio_path: str) -> dict:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        initial_state = {
            "audio_path": audio_path,
            "transcription": "",
            "subtitles": {},
            "answer": "",
            "all_actions": []
        }

        final_state = await self.graph_runnable.ainvoke(initial_state)
        return {
            "transcription": final_state["transcription"],
            "subtitles": final_state["subtitles"],
            "answer": final_state["answer"]
        }

    async def query(self, question: str, history: List[tuple] = None, prompt_template: str = None) -> str:
        # This method can be used to answer questions about the transcription
        if not history:
            history = []

        prompt = get_prompt_template(prompt_template, self.config) if prompt_template else ChatPromptTemplate.from_template(
            "Answer the following question about the transcription: {question}"
        )

        chain = (
            prompt
            | self.transcribe_model.bind(temperature=self.temperature, max_tokens=self.max_tokens)
            | StrOutputParser()
        )

        response = await chain.ainvoke({
            "question": question,
            "history": _format_history(history) if history else []
        })

        return response if response is not None else "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

    def save_subtitles(self, subtitles: dict, base_path: str):
        for format, content in subtitles.items():
            file_path = f"{base_path}.{format}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        return f"Subtitles saved to {base_path}.[srt/vtt/ass/json]"