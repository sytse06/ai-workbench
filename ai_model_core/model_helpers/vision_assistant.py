# model_helpers/vision_assistant.py
# Standard library imports
from typing import List, AsyncGenerator, Union, Tuple
import base64
from io import BytesIO
import os
import logging
import asyncio

# Third-party imports
from langchain_community.chat_models import ChatOllama, ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from PIL import Image

# Local imports
from ..shared_utils.factory import get_model
from ..config.settings import load_config
from ..config.credentials import get_api_key, load_credentials

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "AI-Workbench/1.0"

logger = logging.getLogger(__name__)

class VisionAssistant:
    def __init__(self, model_choice: str, **kwargs):
        self.model = get_model(model_choice, **kwargs)
        self.model_choice = model_choice
        self.config = load_config()
        print(f"Initialized model: {type(self.model)}")

    def update_model(self, model_choice: str, **kwargs):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice, **kwargs)
            self.model_choice = model_choice
    
    def _format_history(self, history: List[Tuple[str, str]]) -> List[BaseMessage]:
        formatted_history = []
        for user_msg, ai_msg in history:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history

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
    
    async def process_image(self, image: Image.Image, question: str, stream: bool = False):
        logger.info(f"Processing image with question: {question}, model: {self.model_choice}")
        if image is None:
            return "Please upload an image first."
        
        try:
            if stream:
                result = [chunk async for chunk in self.image_chat(image, question, stream=True)]
            else:
                result = [chunk async for chunk in self.image_chat(image, question, stream=False)]
            
            return result
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"An error occurred while processing the image: {str(e)}"

    async def image_chat(self, image: Image.Image, question: str, stream: bool = False, image_format: str = "PNG") -> AsyncGenerator[str, None]:
        """
        Sends an image and a question to the model and retrieves the response.

        :param image: The image to send.
        :param question: The question to ask.
        :param stream: Whether to stream the response.
        :param image_format: The format to save the image as. Default is "PNG".
        :return: An async generator yielding responses from the model.
        """
        image_b64 = self._convert_to_base64(image, format=image_format)
        if isinstance(self.model, (ChatOpenAI, ChatOllama, ChatAnthropic)):
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
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        try:
            if stream:
                async for chunk in self.model.astream(messages):
                    yield chunk.content if isinstance(chunk.content, str) else chunk.content[0].text
            else:
                response = await self.model.ainvoke(messages)
                yield response.content if isinstance(response.content, str) else response.content[0].text
        except Exception as e:
            yield f"An error occurred: {str(e)}"
            logger.error(f"Error during image chat invocation: {e}")
            logger.error("Full traceback:", exc_info=True)