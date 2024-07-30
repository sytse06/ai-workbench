# ai_model_interface/models/anthropic.py
from ..base import BaseAIModel, Message
from typing import List, Union, Any
from anthropic import AsyncAnthropic
from pydantic import Field, ConfigDict
from ..config.credentials import get_api_key

class AnthropicModel(BaseAIModel):
    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = AsyncAnthropic(api_key=get_api_key('anthropic'))

    async def chat(self, message: str, history: List[tuple[str, str]], stream: bool = False) -> Union[str, Any]:
        messages = self._format_history(history)
        messages.append({"role": "user", "content": message})
        return await self._anthropic_chat(messages, stream)

    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> Union[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        return await self._anthropic_chat(messages, stream)

    async def image_chat(self, image: bytes, question: str) -> str:
        image_b64 = self._convert_to_base64(image)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
            ]}
        ]
        return await self._anthropic_chat(messages, stream=False)

    async def _anthropic_chat(self, messages: List[dict], stream: bool) -> Union[str, Any]:
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                messages=messages,
                stream=stream
            )
            if stream:
                return self._stream_response(response)
            else:
                return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    async def _stream_response(self, stream):
        partial_message = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    def _format_history(self, history: List[tuple[str, str]]) -> List[dict]:
        return [
            {"role": "user", "content": user_msg}
            if i % 2 == 0 else
            {"role": "assistant", "content": assistant_msg}
            for i, (user_msg, assistant_msg) in enumerate(history)
        ]

    def _convert_to_base64(self, image: bytes) -> str:
        import base64
        return base64.b64encode(image).decode('utf-8')