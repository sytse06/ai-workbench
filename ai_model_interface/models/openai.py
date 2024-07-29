# ai_model_interface/models/openai.py
from ..base import BaseAIModel, Message
from typing import List, Union, Any
from openai import AsyncOpenAI
from pydantic import Field

class OpenAIModel(BaseAIModel):
    os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']
    
    async def chat(self, message: str, history: List[tuple[str, str]], stream: bool = False) -> Union[str, Any]:
        messages = self._format_history(history)
        messages.append({"role": "user", "content": message})
        return await self._openai_chat(messages, stream)

    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> Union[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        return await self._openai_chat(messages, stream)

    async def image_chat(self, image: bytes, question: str) -> str:
        image_b64 = self._convert_to_base64(image)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ]
        return await self._openai_chat(messages, stream=False)

    async def _openai_chat(self, messages: List[dict], stream: bool) -> Union[str, Any]:
        client = AsyncOpenAI(api_key=self.api_key)
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream
            )
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
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