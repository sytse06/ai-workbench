# ai_model_core/model_helpers/chat_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging

#Local imports
from .chat_assistant import ChatAssistant
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.message_types import GradioMessage

logger = logging.getLogger(__name__)

class ChatAssistantUI(ChatAssistant, BaseAssistantUI):
    def __init__(self, assistant_instance=None, *args, **kwargs):
        if assistant_instance is None:
            assistant_instance = ChatAssistant(*args, **kwargs)
        
        super().__init__(assistant_instance)
        self.chat_assistant = assistant_instance

    async def process_gradio_chat(
        self,
        message: Union[str, Dict],
        history: List[Dict],
        model_choice: str,
        temperature: float,
        max_tokens: int,
        files: Optional[List[gr.File]] = None,
        use_context: bool = True,
        history_flag: bool = True,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Process a chat message from Gradio UI, handles model updates,
        message formatting, and calls the chat method.
        """
        try:
            await self.chat_assistant.update_model(model_choice)
            self.chat_assistant.set_temperature(temperature)
            self.chat_assistant.set_max_tokens(max_tokens)

            # Convert to GradioMessage
            if isinstance(message, str):
                gradio_message = GradioMessage(role="user", content=message)
            elif isinstance(message, dict):
                gradio_message = GradioMessage(role="user", content=message.get("content", ""))
            else:
                gradio_message = GradioMessage(role="user", content=str(message))
                
            # Convert to LangChain format
            langchain_message = await self.message_processor.gradio_to_langchain(gradio_message, files)
            
            # Format history
            langchain_history = []
            if history and history_flag:
                for h in history:
                    if isinstance(h, dict) and "role" in h:
                        h_message = GradioMessage(role=h["role"], content=h.get("content", ""))
                        if h["role"] == "user":
                            langchain_history.append(
                                await self.message_processor.gradio_to_langchain(h_message)
                            )
                        else:
                            langchain_history.append(
                                self.message_processor.format_assistant_message(h_message.content)
                            )

            # Call the chat method directly on the assistant instance
            async for response in self.chat_assistant.chat(
                message=langchain_message,
                history=langchain_history,
                prompt_info=prompt_info,
                language_choice=language_choice,
                history_flag=history_flag,
                stream=True,
                use_context=use_context
            ):
                yield self.format_gradio_response(response)

        except Exception as e:
            logger.error(f"Gradio chat processing error: {str(e)}")
            yield {"role": "assistant", "content": f"An error occurred: {str(e)}"}