from ..base import BaseAIModel, Message
from ..utils import format_prompt
from typing import List, AsyncGenerator, Union, Any, Tuple
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from PIL import Image
from io import BytesIO
import base64
import logging
from pydantic import Field, BaseModel, ConfigDict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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

    async def prompt(self, message: str, system_prompt: str, prompt_info: str, stream: bool = False) -> AsyncGenerator[str, None]:
        formatted_prompt = format_prompt(system_prompt, message, prompt_info)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_prompt)
        ]

        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            response = await self.model.ainvoke(messages)
            yield response.content
        
    def _convert_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Converts a PIL Image to a base64 encoded string.

        :param image: PIL Image to be encoded
        :param format: Image format to save as, default is "PNG"
        :return: Base64 encoded string of the image
        """
        try:
            buffered = BytesIO()
            image.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""

    async def image_chat(self, image: Image.Image, question: str, stream: bool = False, image_format: str = "PNG") -> AsyncGenerator[str, None]:
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

class OllamaRunnable(Runnable):
    def __init__(self, model: OllamaModel):
        self.model = model

    async def arun(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> AsyncGenerator[str, None]:
        if isinstance(messages, list) and all(isinstance(msg, (HumanMessage, AIMessage, SystemMessage)) for msg in messages):
            async for chunk in self.model.chat(messages[0].content, []):
                yield chunk
        else:
            raise ValueError("Input to `arun` must be a list of HumanMessage, AIMessage, or SystemMessage instances.")