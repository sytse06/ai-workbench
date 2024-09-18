# model_helpers/chat_assistant.py
import logging
from typing import List, Generator
from langchain.schema import HumanMessage, AIMessage
from ai_model_core import get_model

logger = logging.getLogger(__name__)

class ChatAssistant:
    def __init__(self, model_choice: str, temperature: float = 0.7, min_tokens: int = None, max_tokens: int = None):
        self.model = get_model(model_choice)
        self.model_choice = model_choice
        self.temperature = temperature
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def update_model(self, model_choice: str):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice)
            self.model_choice = model_choice

    def _format_history(self, history: List[tuple[str, str]]) -> List[HumanMessage | AIMessage]:
        formatted_history = []
        for human, ai in history:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history

    async def chat(self, message: str, history: List[tuple[str, str]], history_flag: bool, stream: bool = False) -> Generator[str, None, None]:
        logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {self.model_choice}")
        logger.info(f"Temperature: {self.temperature}, Min tokens: {self.min_tokens}, Max tokens: {self.max_tokens}")
        
        messages = []
        if history_flag:
            messages.extend(self._format_history(history))
        messages.append(HumanMessage(content=message))
        
        # Configure the model with the current settings
        self.model = self.model.bind(
            temperature=self.temperature,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens
        )
        
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            result = await self.model.agenerate([messages])
            yield result.generations[0][0].text

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_min_tokens(self, min_tokens: int):
        self.min_tokens = min_tokens

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens