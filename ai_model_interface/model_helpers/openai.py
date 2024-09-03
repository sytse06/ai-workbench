# ai_model_interface/models/openai.py
from ..base import BaseAIModel, Message
from typing import List, AsyncGenerator, Union
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import Field, ConfigDict
import logging
from PIL import Image
from io import BytesIO
import base64
from ..config.credentials import get_api_key
from ..utils import format_prompt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class OpenAIModel(BaseAIModel):
    model: ChatOpenAI = None
    api_key: str = Field(default_factory=lambda: get_api_key('openai'))

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key
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
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def image_chat(self, image: Image.Image, question: str, stream: bool = False, image_format: str = "PNG") -> AsyncGenerator[str, None]:
        image_b64 = self._convert_to_base64(image, format=image_format)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format.lower()};base64,{image_b64}"
                        }
                    }
                ]
            }
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