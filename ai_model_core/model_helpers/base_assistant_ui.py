# ai_model_core/model_helpers/base_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator, Any
import gradio as gr
import logging
from ..shared_utils.message_processing import MessageProcessor

logger = logging.getLogger(__name__)

class BaseAssistantUI:
    def __init__(self, assistant_instance: Any):
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

            formatted_message = self.message_processor.format_user_message(message, files)
            formatted_history = self._format_history(history) if history and history_flag else []

            async for response in self.assistant.process(
                message=formatted_message,
                history=formatted_history,
                prompt_info=prompt_info,
                language_choice=language_choice,
                history_flag=history_flag,
                stream=True,
                use_context=use_context,
                **kwargs
            ):
                yield self.format_gradio_response(response)

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