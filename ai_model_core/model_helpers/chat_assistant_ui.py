# ai_model_core/model_helpers/chat_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging
import sys

#Local imports
from .chat_assistant import ChatAssistant
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.message_types import GradioMessage

logger = logging.getLogger(__name__)

class ChatAssistantUI:
    """
    UI class for Chat Assistant that composes BaseAssistantUI and 
    assistant functionality through composition.
    """
    def __init__(self, assistant_obj=None, *args, **kwargs):
        # Initialize message processor
        self.message_processor = MessageProcessor()
        
        # Store the assistant object without type checking
        self._assistant = assistant_obj
        
        # Create the UI helper
        if self._assistant is not None:
            self.ui = BaseAssistantUI(self._assistant)
        else:
            # No UI helper if no assistant
            self.ui = None
            logger.warning("ChatAssistantUI initialized without an assistant")

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
            # Update assistant model if needed
            if hasattr(self._assistant, 'update_model'):
                await self._assistant.update_model(model_choice)
            
            # Set parameters if methods exist
            if hasattr(self._assistant, 'set_temperature'):
                self._assistant.set_temperature(temperature)
                
            if hasattr(self._assistant, 'set_max_tokens'):
                self._assistant.set_max_tokens(max_tokens)

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

            # Call the chat method on the assistant
            if hasattr(self._assistant, 'chat'):
                async for response in self._assistant.chat(
                    message=gradio_message,
                    history=langchain_history,
                    prompt_info=prompt_info,
                    language_choice=language_choice,
                    history_flag=history_flag,
                    stream=True,
                    use_context=use_context
                ):
                    # Use UI's formatter if available, otherwise format directly
                    if self.ui:
                        yield self.ui.format_gradio_response(response)
                    else:
                        yield self.format_gradio_response(response)
            else:
                yield {"role": "assistant", "content": "The assistant does not have a chat method."}

        except Exception as e:
            logger.error(f"Gradio chat processing error: {str(e)}")
            yield {"role": "assistant", "content": f"An error occurred: {str(e)}"}
            
    def format_gradio_response(self, response):
        """Format response for Gradio when UI is not available"""
        if isinstance(response, str):
            return {"role": "assistant", "content": response}
        elif isinstance(response, dict) and "content" in response:
            role = response.get("role", "assistant")
            return {"role": role, "content": response["content"]}
        else:
            return {"role": "assistant", "content": str(response)}
        
    async def process_gradio_message(self, *args, **kwargs):
        """Delegate to UI component if available"""
        if self.ui:
            async for response in self.ui.process_gradio_message(*args, **kwargs):
                yield response
        else:
            yield {"role": "assistant", "content": "UI component not initialized."}