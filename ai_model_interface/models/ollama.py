from ..base import BaseAIModel, Message
from typing import List, AsyncGenerator, Union, Any, Tuple
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from PIL import Image
from io import BytesIO
import base64
from pydantic import Field, BaseModel, ConfigDict

class OllamaModel(BaseAIModel):
    base_url: str = "http://localhost:11434"
    model: ChatOllama = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url
        )

    async def chat(self, message: str, history: List[tuple[str, str]], stream: bool = False) -> AsyncGenerator[str, None]:
        messages = self._format_history(history)
        messages.append(HumanMessage(content=message))
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            response = await self.model.ainvoke(messages)
            yield response.content

    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> AsyncGenerator[str, None]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            response = await self.model.ainvoke(messages)
            yield response.content

    def _convert_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def image_chat(self, image: bytes, question: str, stream: bool = False, image_format: str = "PNG") -> AsyncGenerator[str, None]:
        image_b64 = self._convert_to_base64(image, format=image_format)
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": f"data:image/{image_format.lower()};base64,{image_b64}"}
                ]
            )
        ]
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            response = await self.model.ainvoke(messages)
            yield response.content

    def _format_history(self, history: List[tuple[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
        formatted_history = []
        for user_msg, ai_msg in history:
            formatted_history.append(HumanMessage(content=user_msg))
            formatted_history.append(AIMessage(content=ai_msg))
        return formatted_history