from langchain_community.chat_models import ChatOpenAI, ChatOllama, ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from ai_model_interface.factory import get_model
from ai_model_interface.config.settings import load_config
from ai_model_interface.config.credentials import get_api_key, load_credentials
from typing import List, AsyncGenerator, Union
from PIL import Image
import base64
from io import BytesIO
import os
import sys
import gradio as gr
import logging
import asyncio
class VisionAssistant:
    def __init__(self, model_choice: str, **kwargs):
        self.model = get_model(model_choice, **kwargs)
        self.model_choice = model_choice
        self.config = load_config()

    def update_model(self, model_choice: str, **kwargs):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice, **kwargs)
            self.model_choice = model_choice

    def _convert_to_base64(self, image: Union[str, Image.Image, bytes], format: str = "PNG") -> str:
        if isinstance(image, str):
            if os.path.isfile(image):
                with open(image, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            else:
                return image  # Assume it's already a base64 string
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError("Unsupported image type")

    def _format_image_content(self, image_b64: str, image_format: str):
        if isinstance(self.model, ChatOpenAI):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format.lower()};base64,{image_b64}"
                }
            }
        else:
            return f"data:image/{image_format.lower()};base64,{image_b64}"

    async def process_image(self, image: Union[str, Image.Image, bytes], question: str, model_choice: str, stream: bool = False) -> AsyncGenerator[str, None]:
        self.update_model(model_choice)
        image_b64 = self._convert_to_base64(image)
        image_content = self._format_image_content(image_b64, "PNG")

        if isinstance(self.model, ChatOpenAI):
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": image_content}
                ])
            ]
        elif isinstance(self.model, (ChatOllama, ChatAnthropic)):
            messages = [
                HumanMessage(content=f"{question}\n\n[IMAGE]{image_content}[/IMAGE]")
            ]
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        try:
            if stream:
                async for chunk in self.model.astream(messages):
                    yield chunk.content
            else:
                response = await self.model.ainvoke(messages)
                yield response.content
        except Exception as e:
            yield f"An error occurred: {str(e)}"