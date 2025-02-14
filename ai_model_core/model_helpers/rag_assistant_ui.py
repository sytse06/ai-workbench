# ai_model_core/model_helpers/rag_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging
from pathlib import Path

# Local imports
from .RAG_assistant import RAGAssistant
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.message_types import GradioMessage
from ..shared_utils.factory import update_model as factory_update_model

logger = logging.getLogger(__name__)

class RAGAssistantUI(RAGAssistant, BaseAssistantUI):
    """
    UI class for RAG Assistant that implements Gradio v5 message structure
    and integrates with the base UI components.
    """
    def __init__(self, *args, **kwargs):
        RAGAssistant.__init__(self, *args, **kwargs)
        BaseAssistantUI.__init__(self)
        self.message_processor = MessageProcessor()

    async def process_gradio_message(
        self,
        message: Union[str, Dict],
        history: List[Dict],
        **kwargs
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Process messages using Gradio v5 format and generate responses.
        """
        try:
            # Extract message text using message processor
            message_text = await self.message_processor.get_message_text(message)
            
            # Process through query method
            async for response in self.query(
                message=message_text,
                history=history,
                stream=True,
                **kwargs
            ):
                if isinstance(response, dict):
                    yield response
                else:
                    yield {"role": "assistant", "content": str(response)}
                    
        except Exception as e:
            logger.error(f"Error in process_gradio_message: {str(e)}")
            yield {
                "role": "assistant",
                "content": f"An error occurred: {str(e)}"
            }

    async def _process_uploaded_files(
        self,
        files: List[gr.File]
    ) -> None:
        """
        Process uploaded files and update the vector store.
        
        Args:
            files: List of Gradio file components
        """
        try:
            file_paths = []
            for file in files:
                if hasattr(file, "name"):
                    file_path = Path(file.name)
                    if file_path.exists():
                        file_paths.append(str(file_path))

            if file_paths:
                result = await self.process_content(
                    url_input="",
                    file_input=file_paths
                )
                logger.info(f"File processing result: {result}")

        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            raise

    async def _format_history(
        self,
        history: List[Dict]
    ) -> List:
        """
        Convert Gradio message history to LangChain format.
        
        Args:
            history: List of message dictionaries in Gradio format
        
        Returns:
            List of messages in LangChain format
        """
        formatted_history = []
        for h in history:
            if isinstance(h, dict) and "role" in h:
                gradio_message = GradioMessage(
                    role=h["role"],
                    content=h.get("content", "")
                )
                langchain_message = await self.message_processor.gradio_to_langchain(
                    gradio_message
                )
                formatted_history.append(langchain_message)
        return formatted_history

    def format_gradio_response(
        self,
        response: Union[str, Dict]
    ) -> Dict[str, str]:
        """
        Format the response in Gradio v5 message structure.
        
        Args:
            response: Response string or dictionary
        
        Returns:
            Dictionary with role and content keys
        """
        if isinstance(response, str):
            return {"role": "assistant", "content": response}
        elif isinstance(response, dict) and "role" in response:
            return response
        elif hasattr(response, "role") and hasattr(response, "content"):
            return {"role": response.role, "content": response.content}
        return {"role": "assistant", "content": str(response)}

    async def update_model(self, model_choice: str) -> None:
        """
        Update the model if a different one is selected.
        
        Args:
            model_choice: Name of the model to use
        """
        try:
            new_model = await factory_update_model(model_choice, self.model_choice)
            if new_model:
                self.model_local = new_model
                self.model_choice = model_choice
                logger.info(f"Model updated to {model_choice}")
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise

    def set_temperature(
        self,
        temperature: float
    ) -> None:
        """
        Set the temperature parameter for generation.
        
        Args:
            temperature: Temperature value between 0 and 1
        """
        self.temperature = temperature

    def set_max_tokens(
        self,
        max_tokens: int
    ) -> None:
        """
        Set the maximum tokens for generation.
        
        Args:
            max_tokens: Maximum number of tokens
        """
        self.max_tokens = max_tokens