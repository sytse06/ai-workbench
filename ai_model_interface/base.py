# ai_model_interface/base.py
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class BaseAIModel(BaseModel, ABC):
    model_name: str

    @abstractmethod
    async def chat(self, message: str, history: List[tuple[str, str]], stream: bool = False) -> Union[str, Any]:
        pass

    @abstractmethod
    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> Union[str, Any]:
        pass

    @abstractmethod
    async def image_chat(self, image: bytes, question: str) -> str:
        pass