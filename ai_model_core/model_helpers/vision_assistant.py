from langchain_community.chat_models import ChatOpenAI, ChatOllama, ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from ai_model_core.factory import get_model
from ai_model_core.config.settings import load_config
from ai_model_core.config.credentials import get_api_key, load_credentials
from typing import List, AsyncGenerator, Union
from PIL import Image
import base64
from io import BytesIO
import os
import logging
import asyncio

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
        """
        Sends an image and a question to the model and retrieves the response.

        :param image: The image to send.
        :param question: The question to ask.
        :param stream: Whether to stream the response.
        :param image_format: The format to save the image as. Default is "PNG".
        :return: An async generator yielding responses from the model.
        """
        image_b64 = self._convert_to_base64(image, format=image_format)
        #image_content = self._format_image_content(image_b64, image_format)

        logger.info(f"Sending image with format: {format}")

        # Handling different model types within the image_chat function
        if isinstance(self.model, ChatOpenAI):
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
        elif isinstance(self.model, ChatOllama) or isinstance(self.model, ChatAnthropic):
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/{image_format.lower()};base64,{image_b64}"}
                    ]
                )
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
            logger.error(f"Error during image chat invocation: {e}")
            logger.error("Full traceback:", exc_info=True)


async def process_image(image: Image.Image, question: str, model_choice: str, stream: bool = False):
    """
	Handles the process of sending an image and a question to the specified model.

	:param image: The image to process.
	:param question: The question to ask.
	:param model_choice: The model to be used.
	:param stream: Whether to stream the response.
	:return: A list of results from the model.
	"""

    logger.info(f"Process image called with question: {question}, model_choice: {model_choice}")

    if image is None:
	        return "Please upload an image first."

	# Initialize the VisionAssistant with the chosen model
    assistant = VisionAssistant(model_choice=model_choice)
     
    logger.info(f"VisionAssistant instantiated with model_choice: {model_choice}")

	# Handle streaming or non-streaming mode using image_chat method
    if stream:
	    result = [chunk async for chunk in assistant.image_chat(image, question, stream=True)]
    else:
	    result = [chunk async for chunk in assistant.image_chat(image, question, stream=False)]
	    
    return result