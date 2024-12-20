# model_helpers/chat_assistant.py
import logging
import os
from typing import List, Generator, Union, Tuple
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from ai_model_core import get_model

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "AI-Workbench/1.0"

logger = logging.getLogger(__name__)

class ChatAssistant:
    def __init__(self, model_choice: str, temperature: float = 0.7, max_tokens: int = None):
        self.model = get_model(model_choice)
        self.model_choice = model_choice
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update_model(self, model_choice: str):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice)
            self.model_choice = model_choice

    def _format_history(self, history: List[Tuple[str, str]]) -> List[BaseMessage]:
        formatted_history = []
        for human, ai in history:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history

    async def chat(
        self, 
        message: str, 
        history: List[Tuple[str, str]], 
        history_flag: bool, 
        stream: bool = False
    ) -> Generator[str, None, None]:
        logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {self.model_choice}")
        logger.info(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
        
        messages = []
        if history_flag:
            messages.extend(self._format_history(history))
        messages.append(HumanMessage(content=message))
        
        # Configure the model with the current settings
        self.model = self.model.bind(
            temperature=self.temperature,
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

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens