# ai_model_interface/base.py
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Tuple, AsyncGenerator
from pydantic import BaseModel, ConfigDict

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]

class BaseAIModel(BaseModel, ABC):
    model_name: str
        
    model_config = ConfigDict(protected_namespaces=())

    @abstractmethod
    async def chat(self, message: str, history: List[Tuple[str, str]], stream: bool = False) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def image_chat(self, image: bytes, question: str) -> AsyncGenerator[str, None]:
        pass