# ai_model_core/model_helpers/chat_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging

#Local imports
from .chat_assistant import ChatAssistant
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor

logger = logging.getLogger(__name__)

class ChatAssistantUI(ChatAssistant, BaseAssistantUI):
    def __init__(self, *args, **kwargs):
        ChatAssistant.__init__(self, *args, **kwargs)
        BaseAssistantUI.__init__(self)

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
        try:
            await self.update_model(model_choice)
            self.set_temperature(temperature)
            self.set_max_tokens(max_tokens)

            formatted_message = self.message_processor.format_user_message(message, files)
            formatted_history = []
            
            if history and history_flag:
                for h in history:
                    if isinstance(h, dict) and "role" in h:
                        if h["role"] == "user":
                            formatted_history.append(
                                self.message_processor.format_user_message(h)
                            )
                        else:
                            formatted_history.append(
                                self.message_processor.format_assistant_message(h["content"])
                            )

            async for response in self.chat(
                message=formatted_message,
                history=formatted_history,
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