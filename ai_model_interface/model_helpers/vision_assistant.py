from langchain.chat_models import ChatOpenAI, ChatOllama, ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from ai_model_interface.factory import get_model
from typing import List, Dict, Any
from PIL import Image
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class VisionAssistant:
    def __init__(self, model_choice: str, **kwargs):
        self.model = get_model(model_choice, **kwargs)
        self.model_choice = model_choice
    
    def update_model(self, model_choice: str, **kwargs):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice, **kwargs)
            self.model_choice = model_choice
    
    def _format_history(self, history: List[tuple[str, str]]) -> List[HumanMessage | AIMessage]:
        formatted_history = []
        for user_msg, ai_msg in history:
            formatted_history.append(HumanMessage(content=user_msg))
            formatted_history.append(AIMessage(content=ai_msg))
        return formatted_history

    def _convert_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _format_image_content(self, image_b64: str, image_format: str):
        if "OpenAI" in self.model_choice:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format.lower()};base64,{image_b64}"
                }
            }
        else:
            return {
                "type": "image_url",
                "image_url": f"data:image/{image_format.lower()};base64,{image_b64}"
            }

    async def image_chat(self, image: Image.Image, question: str, stream: bool = False, image_format: str = "PNG"):
        image_b64 = self._convert_to_base64(image, format=image_format)
        image_content = self._format_image_content(image_b64, image_format)
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    image_content
                ]
            )
        ]
        
        try:
            if stream:
                async for chunk in self.model.astream(messages):
                    yield chunk.content
            else:
                response = await self.model.ainvoke(messages)
                yield response.content
        except Exception as e:
            logger.error(f"Error in image_chat: {e}")
            yield f"An error occurred: {str(e)}"