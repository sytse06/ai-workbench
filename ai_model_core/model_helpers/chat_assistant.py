# model_helpers/chat_assistant.py
import logging
from typing import List, Generator
from langchain.schema import HumanMessage, AIMessage
from ai_model_core import get_model

logger = logging.getLogger(__name__)

class ChatAssistant:
    def __init__(self, model_choice: str):
        self.model = get_model(model_choice)
        self.model_choice = model_choice

    def update_model(self, model_choice: str):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice)
            self.model_choice = model_choice
            
    def format_conversation_history(self, history):
        return "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])

    def _format_history(self, history: List[tuple[str, str]]) -> List[HumanMessage | AIMessage]:
        formatted_history = []
        for human, ai in history:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history

    async def chat(self, message: str, history: List[tuple[str, str]], history_flag: bool, stream: bool = False) -> Generator[str, None, None]:
        logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {self.model_choice}")
        
        messages = []
        if history_flag:
            messages.extend(self._format_history(history))
        messages.append(HumanMessage(content=message))
        
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            result = await self.model.agenerate([messages])
            yield result.generations[0][0].text