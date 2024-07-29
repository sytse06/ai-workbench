# ai_model_interface/models/ollama.py
from ..base import BaseAIModel, Message
from typing import List, Union, Any
import ollama

class OllamaModel(BaseAIModel):
    base_url: str = Field(default="http://localhost:11434")

    async def chat(self, message: str, history: List[tuple[str, str]], stream: bool = False) -> Union[str, Any]:
        messages = self._format_history(history)
        messages.append(Message(role="user", content=message))
        return await self._ollama_chat(messages, stream)

    async def prompt(self, message: str, system_prompt: str, stream: bool = False) -> Union[str, Any]:
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=message)
        ]
        return await self._ollama_chat(messages, stream)

    async def image_chat(self, image: bytes, question: str) -> str:
        image_b64 = self._convert_to_base64(image)
        messages = [
            Message(role="user", content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            ])
        ]
        return await self._ollama_chat(messages, stream=False)

    async def _ollama_chat(self, messages: List[Message], stream: bool) -> Union[str, Any]:
        try:
            response = await ollama.chat(
                model=self.model_name,
                messages=[m.dict() for m in messages],
                stream=stream
            )
            if stream:
                return self._stream_response(response)
            else:
                return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_response(self, stream):
        partial_message = ""
        for chunk in stream:
            if chunk['message']['content']:
                partial_message += chunk['message']['content']
                yield partial_message

    def _format_history(self, history: List[tuple[str, str]]) -> List[Message]:
        return [
            Message(role="user", content=user_msg)
            if i % 2 == 0 else
            Message(role="assistant", content=assistant_msg)
            for i, (user_msg, assistant_msg) in enumerate(history)
        ]

    def _convert_to_base64(self, image: bytes) -> str:
        import base64
        return base64.b64encode(image).decode('utf-8')