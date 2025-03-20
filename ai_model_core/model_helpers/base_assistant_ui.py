# ai_model_core/model_helpers/base_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator, Any
import gradio as gr
import logging

#local imports
from ..shared_utils.message_types import (
    BaseMessageProcessor,
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from ..shared_utils.message_processing import MessageProcessor

logger = logging.getLogger(__name__)

class BaseAssistantUI:
    def __init__(self, assistant_instance: Any):
        """
        Initialize BaseAssistantUI with an assistant instance.
        
        Args:
            assistant_instance: An instance of a chat assistant class
                that implements methods like get_model and update_model, set_temperature,
                set_max_tokens, and chat.
        """
        self.assistant = assistant_instance
        self.message_processor = MessageProcessor()

    async def process_gradio_message(
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
        language_choice: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, str], None]:
        try:
            await self.assistant.update_model(model_choice)
            self.assistant.set_temperature(temperature)
            self.assistant.set_max_tokens(max_tokens)
            
            # Convert string message to GradioMessage format
            if isinstance(message, str):
                gradio_message = GradioMessage(role="user", content=message)
            elif isinstance(message, dict):
                gradio_message = GradioMessage(role="user", content=message.get("content", ""))
            else:
                gradio_message = GradioMessage(role="user", content=str(message))

            # Create GradioMessage to LangChain format
            langchain_message = await self.message_processor.gradio_to_langchain(gradio_message, files)

            # Format history properly
            langchain_history = []
            if history and history_flag:
                for h in history:
                    if isinstance(h, str):
                        h = {"role": "user", "content": h}
                    if isinstance(h, dict) and "role" in h:
                        msg = GradioMessage(role=h["role"], content=h.get("content", ""))
                        if h["role"] == "user":
                            langchain_history.append(
                                await self.message_processor.gradio_to_langchain(msg)
                            )
                        else:
                            langchain_history.append(
                                self.message_processor.format_assistant_message(msg.content)
                            )

            async for response in self.assistant.chat(
                message=langchain_message,
                history=langchain_history,
                prompt_info=prompt_info,
                language_choice=language_choice,
                history_flag=history_flag,
                stream=True,
                use_context=use_context
            ):
                # Handle string or BaseMessage response types
                if isinstance(response, str):
                    yield {"role": "assistant", "content": response}
                else:
                    gradio_message = self.message_processor.langchain_to_gradio(response)
                    yield {"role": gradio_message.role, "content": gradio_message.content}

        except Exception as e:
            logger.error(f"Gradio processing error: {str(e)}")
            yield {"role": "assistant", "content": f"An error occurred: {str(e)}"}

    def _format_history(self, history: List[Dict]) -> List:
        formatted_history = []
        for h in history:
            if isinstance(h, dict) and "role" in h:
                if h["role"] == "user":
                    formatted_history.append(self.message_processor.format_user_message(h))
                else:
                    formatted_history.append(self.message_processor.format_assistant_message(h["content"]))
        return formatted_history

    def format_gradio_response(self, response) -> Dict[str, str]:
        if isinstance(response, str):
            return {"role": "assistant", "content": response}
        elif isinstance(response, dict) and "role" in response:
            return response
        elif hasattr(response, "role") and hasattr(response, "content"):
            return {"role": response.role, "content": response.content}
        return {"role": "assistant", "content": str(response)}